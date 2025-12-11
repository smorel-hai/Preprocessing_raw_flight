import json
from pathlib import Path
import numpy as np
import shutil
from yaml import safe_load
from utils.utils import compute_intersection, convert_wgs84_to_web_mercator, add_percentage_margin_to_bbox, is_bbox_inside, compute_enclosing_box, get_union_of_bboxes, transfer_skip_existing_names, delete_folder
from geopy.geocoders import Nominatim
from retreieve_satellite_image_depending_on_coord import download_tiles, merge_tiles_to_geotiff, get_bounding_box
from crotalinae_pg.data_preprocessing.raw.video_processor import VideoProcessor
from crotalinae_pg.data_preprocessing.raw.utils.devices import load_device_config
from retrieve_position_from_UAV_view import compute_frames_fov
from pruning_tile import prune_redundant_areas_with_rotation
from extract_satellite_tile_from_drone_view import get_best_tile_for_fov, save_tile_to_disk


def get_city_from_point(lat, lon):
    geolocator = Nominatim(user_agent="my_geo_app")
    location = geolocator.reverse((lat, lon), language='en')
    address = location.raw['address']
    return address.get('city') or address.get('town') or address.get('village')


class Region:
    def __init__(self, bbox):
        #  Assure to order the bounding box (nw, se)
        self.bbox = get_bounding_box(bbox)
        self._get_name()
        self.tiff_path = None
        self.temp_dir = None

    def _get_name(self):
        self.barycenter = np.mean(np.array(self.bbox), axis=0)
        self.name = get_city_from_point(*self.barycenter)

    def download_corresponding_tiff_file(self, save_path, temp_folder, api_key, margin, zoom):
        temp_folder = Path(temp_folder) / self.name
        temp_folder.mkdir(exist_ok=True, parents=True)
        #  If we want to have some extra margin around the salellite views, we need to have some margin
        bbox_with_margin = add_percentage_margin_to_bbox(self.bbox[0], self.bbox[1], margin)

        download_tiles(bbox_with_margin[0], bbox_with_margin[1], api_key, temp_folder, zoom=zoom)
        merge_tiles_to_geotiff(temp_folder, save_path, zoom)

        #  TODO: better deal with that if we load the dataset and do an update of a region : need the tiff_path and temp_dir....
        #  So, at loading, need to retrieve the correct path in the corresponding Region
        self.tiff_path = save_path
        self.temp_dir = temp_folder

    def update_bbox(self, new_bbox):
        self.bbox = new_bbox


class Dataset:
    def __init__(self, root_dir: str, video_dir: str, config_file: str, margin: float, zoom: int, api_key: str,
                 enable_region_merge: bool = True, iou_thrshold: float = 0.5, angle_threshold: float = 15.0,
                 guide_match_iou_threshold: float = 0.9):
        self.root_dir = Path(root_dir)
        self.video_dir = Path(video_dir)
        self.config_file = config_file

        self.region_dict = {}
        self.updated_region = set()  #  Usefull for knowing the region that has been updated, that needs to download the global tiff
        self.metadata_json_path = self.root_dir / 'metadata'
        self.load_json()

        self.margin = margin
        self.zoom = zoom
        self.api_key = api_key

        self.iou_threshold = iou_thrshold
        self.angle_threshold = angle_threshold
        self.guide_match_iou_threshold = guide_match_iou_threshold

        self.enable_region_merge = enable_region_merge

    def load_json(self):
        if not self.metadata_json_path.exists():
            self.metadata = {}
        else:
            # TODO: instantier la liste des Regions lors du load
            with open(self.metadata_json_path) as f:
                self.metadata = json.load(f)

    def save_metadata_json(self):
        with open(self.metadata_json_path, "w") as file:
            json.dump(self.metadata, file)

    def merge_region(self, regions_to_merge_name):
        list_region_bbox = [self.region_dict[region_name].bbox for region_name in regions_to_merge_name]
        merge_bbox = get_union_of_bboxes(list_region_bbox)
        # Create new region acourding to the new bbox
        merge_region_name = self.create_new_region(merge_bbox)
        #  Now, move all the old files to the new one corresponding and correct metadata
        self.metadata[merge_region_name]['zones'] = {}
        for former_name in regions_to_merge_name:
            transfer_skip_existing_names(self.root_dir / former_name, self.root_dir / merge_region_name)
            delete_folder(self.root_dir / former_name)
            #  update metadata zones with the keys that the new region has not : all the key
            self.metadata[merge_region_name]['zones'] |= self.metadata[former_name]['zones']
            #  Remove the former_name from the region dict and the updated_region
            self.region_dict.pop(former_name)
            self.updated_region.discard(former_name)
        return merge_region_name

    def search_corresponding_region(self, bbox_candidate):
        corresponding_region_candidates = []
        for region_name, region in self.region_dict.items():
            bbox_region = region.bbox
            bbox_intersection = compute_intersection(bbox_candidate, bbox_region)
            if bbox_intersection.max() > 0:
                corresponding_region_candidates.append(region_name)

        if len(corresponding_region_candidates) == 0:
            name_region_in_dict = self.create_new_region(bbox_candidate)
            return name_region_in_dict

        elif len(corresponding_region_candidates) == 1:
            corresponding_region_name = corresponding_region_candidates[0]
            corresponding_region = self.region_dict[corresponding_region_name]
            #  If we have a bbox that is not totally inside the Region, we need to update the region
            increases_region = is_bbox_inside(bbox_candidate, corresponding_region.bbox)

            if increases_region:
                # We increase the region by computing the bigger bouding box around the union of bbox
                region_bbox_updated = compute_enclosing_box(bbox_candidate, corresponding_region.bbox)
                #  Need to update the region
                corresponding_region.update_bbox(region_bbox_updated, self.api_key, self.margin, self.zoom)
                #  Updated it in the region_dict of the Dataset
                self.region_dict[corresponding_region_name] = corresponding_region
                self.metadata[corresponding_region_name]['bbox'] = region_bbox_updated
                self.updated_region.add(region_bbox_updated)
            return corresponding_region_name
        else:
            if not self.enable_region_merge:
                raise ValueError('Merging Region is not allowed despite having a new area that should require it')
            return self.merge_region(corresponding_region_candidates)

    def create_new_region(self, bbox_region):
        new_region = Region(bbox_region)
        name_region = new_region.name
        idx = 0
        name_region_in_dict = name_region + f'_{idx}'
        while (name_region_in_dict in self.region_dict.keys()):
            idx += 1
            name_region_in_dict = name_region + f'_{idx}'

        #  Add new region name to tracking objets
        self.region_dict[name_region_in_dict] = new_region
        self.updated_region.add(name_region_in_dict)

        #  Update metadata_json with the new region
        self.metadata[name_region_in_dict] = {'name': name_region, 'bbox': new_region.bbox, "zones": {}}
        return name_region_in_dict

    def download_global_tiff_updated_region(self):
        for region_name in self.updated_region:
            #  Download the region tiff
            save_path = self.root_dir / region_name / 'satellite' / 'Region_view.tiff'
            temp_dir = self.root_dir / region_name / 'satellite' / '.tmp'
            self.region_dict[region_name].download_corresponding_tiff_file(
                save_path, temp_dir, self.api_key, self.margin, self.zoom)

    def retrieve_all_zone_from_region(self, selected_region):
        zone_bbox_list = []
        zone_name_list = []
        for zone_name, zone_val in self.metadata[selected_region]['zones'].items():
            zone_bbox_list.append(zone_val["bbox"])
            zone_name_list.append(zone_name)
        return zone_bbox_list, zone_name_list

    def extract_candidate_frames(self, video_path, output_root):
        """
        Extracts frames from the video based on the delta_frames configuration.
        """

        print(f"      Input: {video_path}")

        # Load configurations
        config_data = safe_load(open(self.config_path))
        device_config = load_device_config(
            config_data["device"]["config_path"],
            config_data["device"]["id"]
        )

        # Initialize processor
        video_processor = VideoProcessor(
            config_data["preprocessing_args"]["camera_destination_name"],
            output_root,
            config_data["preprocessing_args"]["clips_computer_args"],
            config_data["preprocessing_args"]["delta_frames"]
        )

        # Run extraction
        video_processor.process(Path(video_path), "frames")

        # Retrieve the dataframe containing telemetry (Lat, Lon, Yaw, Pitch, etc.)
        telemetry_metadata = video_processor.frame_saver.metadata

        print(f"      Extracted {len(telemetry_metadata)} frames.")
        return device_config, telemetry_metadata

    def create_zone(self, bbox, region_name):

        corresponding_zone_name = ?
        #  Initialize new zone in metadata
        self.metadata[region_name]['zones'][corresponding_zone_name]['drone'] = {}
        self.metadata[region_name]['zones'][corresponding_zone_name]['satellite'] = {}

    def process_video(self, video_path, fly_name):
        # --- Step 1: Extract Frames from Video ---
        print(f"\n[1/6] Starting Video Processing...")
        tmp_dir = self.root_dir / '.tmp' / Path(video_path).stem
        device_config, frames_metadata = self.extract_candidate_frames(video_path, tmp_dir)

        # --- Step 2: Prepare Camera Geometry ---
        print(f"\n[2/6] Calculating Field of View (FOV) for all frames...")

        # Get intrinsic matrix (Camera lens properties)
        camera_intrinsics = device_config.cameras.get('EO').intrinsic_settings.K_coefs
        img_width, img_height = device_config.cameras.get('EO').intrinsic_settings.image_size
        fov_wgs84_list, rotation_matrix_list, global_bouding_box = compute_frames_fov(
            frames_metadata, img_width, img_height, camera_intrinsics)

        # Store results back in dataframe
        frames_metadata['fov_wgs84'] = fov_wgs84_list

        # Convert FOV coordinates to Web Mercator (Meters) for accurate area calculation
        frames_metadata['fov_mercator'] = [convert_wgs84_to_web_mercator(
            coords) for coords in frames_metadata['fov_wgs84']]

        # --- Step 3: Download Satellite Imagery ---
        print(f"\n[3/6] Retrieve corresponding Region")
        # Calculate the North-West and South-East corners for the tile downloader
        nw_corner, se_corner = get_bounding_box(global_bouding_box)
        corresponding_region_name = self.search_corresponding_region([nw_corner, se_corner])

        # --- Step 4: Prune Redundant Frames ---
        print(f"\n[4/6] Pruning Redundant Frames...")

        guide_coords_list_gps, guide_coords_zone_name = self.retrieve_all_zone_from_region(corresponding_region_name)
        guide_coords_list_mercator = [convert_wgs84_to_web_mercator(coords) for coords in guide_coords_list_gps]

        guided_matches_output = prune_redundant_areas_with_rotation(
            frames_metadata['fov_mercator'],
            rotation_matrix_list,
            guide_coords_list=guide_coords_list_mercator,
            max_areas_to_keep=len(frames_metadata),  # Initially allow all, let threshold decide
            iou_threshold=self.iou_threshold,
            angle_threshold_degrees=self.angle_threshold
        )

        # --- Step 5: Save Filtered Results ---
        print(f"\n[5/6] Dealing with selected Frames ...")

        working_dir = self.root_dir / corresponding_region_name

        save_frame_dir = working_dir / 'drone' / fly_name
        save_frame_dir.mkdir(exist_ok=True, parents=True)

        for frame_indice, corresponding_zone_idx in guided_matches_output.items():
            if corresponding_zone_idx is None:

            else:
                corresponding_zone_name = guide_coords_zone_name[corresponding_zone_idx]

            frame_info = frames_metadata.iloc[frame_indice]
            frame_info_dict = frame_info.to_dict()
            frame_filename = frame_info.index.values[0]

            source_path = tmp_dir / frame_filename
            dest_path = save_frame_dir / frame_filename
            if source_path.exists():
                shutil.copy(source_path, dest_path)
            else:
                print(f"      Warning: Source file missing {source_path}")

            id_drone = dest_path.stem
            info_image_dict = {'image_path': str(dest_path),
                               'fly_id': fly_name}
            info_image_dict |= frame_info_dict
            self.metadata[corresponding_region_name]['zones'][corresponding_zone_name]['drone'][id_drone] = info_image_dict
        # --- Step 6: Extract Matching Satellite Tiles ---
        print(f"\n[6/6] Generating new zones from the remaining selected frames")

        extracted_tiles_dir = working_dir / 'satellite' / 'extracted_tiles'
        extracted_tiles_dir.mkdir(exist_ok=True, parents=True)

        # Iterate over the kept frames
        count = 0
        for fov_coords, media_name in zip(filtered_frames_metadata['fov_mercator'], filtered_frames_metadata.index.values):
            media_stem = Path(media_name).stem  # Remove extension (e.g., .jpg)

            # Crop the big Tiff to match this specific frame's view
            tile_result = get_best_tile_for_fov(working_dir / 'satellite' / 'Region_view.tiff', fov_coords)

            if tile_result:
                save_tile_to_disk(tile_result, str(extracted_tiles_dir / f'{media_stem}_sat.tiff'))
                count += 1

        print(f"      Successfully saved {count} satellite crops.")
