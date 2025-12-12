"""Dataset management module for drone video preprocessing pipeline.

This module provides classes and functions to manage the extraction, processing, and organization
of drone video data into a structured dataset with corresponding satellite imagery.

Key Components:
    - Region: Represents a geographical region with satellite imagery
    - Dataset: Main class managing the entire preprocessing pipeline
"""

import json
from pathlib import Path
import shutil
from datetime import datetime
from yaml import safe_load

# Internal utilities
from utils.utils import convert_wgs84_to_mercator, transfer_skip_existing_names, delete_folder, find_values_gen
from utils.oriented_bbox import oriented_bbox, get_union_of_bboxes

# External dependencies
import reverse_geocoder as rg

# Processing modules
from retreieve_satellite_image_depending_on_coord import download_tiles, merge_tiles_to_geotiff
from utils.extract_filtered_frames import run_pipeline as filter_video
from utils.devices import load_device_config
from retrieve_position_from_UAV_view import compute_frames_fov
from pruning_tile import prune_redundant_areas_with_rotation
from extract_satellite_tile_from_drone_view import get_best_tile_for_fov, save_tile_to_disk


def get_city_from_point(lat: float, lon: float) -> str:
    """Get the city name from GPS coordinates using reverse geocoding.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        City or town name at the given coordinates
    """
    results = rg.search((lat, lon))
    return results[0]['name']


class Region:
    """Represents a geographical region for drone data processing.

    A Region manages a specific geographical area including its bounding box,
    name (derived from location), and associated satellite imagery.

    Attributes:
        zone: Oriented bounding box defining the region's boundaries
        barycenter: Center point (lat, lon) of the region
        name: City/town name at the region's center
    """

    def __init__(self, zone: oriented_bbox):
        """Initialize a Region with a geographical bounding box.

        Args:
            zone: Oriented bounding box defining the region
        """
        self.zone = zone
        self._get_name()

    def _get_name(self) -> None:
        """Set the region name based on its center coordinates."""
        self.barycenter = self.zone.center
        self.name = get_city_from_point(*self.barycenter)

    def download_corresponding_tiff_file(self, save_path: Path, temp_folder: Path, api_key: str,
                                         margin: float, zoom: int, api_name: str = 'MapTiler') -> None:
        """Download satellite imagery for this region as a GeoTIFF file.

        Args:
            save_path: Path where the merged GeoTIFF will be saved
            temp_folder: Temporary directory for downloading individual tiles
            api_key: API key for the satellite imagery service
            margin: Fractional margin to add around the region (e.g., 0.1 = 10%)
            zoom: Zoom level for satellite tiles
            api_name: Name of the satellite API service (default: 'MapTiler')
        """
        temp_folder = Path(temp_folder) / self.name
        temp_folder.mkdir(exist_ok=True, parents=True)

        download_tiles(self.zone.nw, self.zone.se, api_key, temp_folder, zoom=zoom)
        merge_tiles_to_geotiff(temp_folder, save_path, zoom)

        historic_download_json_path = save_path.parent / 'download_historic.json'

        if not historic_download_json_path.exists():
            historic_download_json = {}
        else:
            with open(historic_download_json_path) as f:
                historic_download_json = json.load(f)

        # Get current date and time
        current_date = datetime.now().strftime("%Y_%m_%d")

        historic_download_json_api = historic_download_json.get(api_name, {})
        historic_download_json_api[current_date] = self.zone.get_mercator_bbox()
        historic_download_json[api_name] = historic_download_json_api
        with open(historic_download_json_path, "w") as file:
            json.dump(historic_download_json, file, indent=4)

    def update_bbox(self, new_zone: oriented_bbox):
        self.zone = new_zone


class Dataset:
    def __init__(self, root_dir: str, video_dir: str, config_file: str, margin: float, zoom: int, api_key: str,
                 enable_region_merge: bool = True, iou_thrshold: float = 0.5, angle_threshold: float = 15.0,
                 guide_match_iou_threshold: float = 0.9, verbose: int = 0):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True, parents=True)
        self.video_dir = Path(video_dir)
        self.config_file = config_file

        self.region_dict = {}
        self.updated_region = set()  #  Usefull for knowing the region that has been updated, that needs to download the global tiff
        self.metadata_json_path = self.root_dir / 'metadata.json'
        self.load_json()

        self.margin = margin
        self.zoom = zoom
        self.api_key = api_key

        self.iou_threshold = iou_thrshold
        self.angle_threshold = angle_threshold
        self.guide_match_iou_threshold = guide_match_iou_threshold

        self.enable_region_merge = enable_region_merge
        self.verbose = verbose

    def init_existing_dataset(self) -> None:
        """Initialize dataset from existing metadata.

        Loads existing regions from metadata and checks which ones need
        satellite imagery updates.
        """
        for key, val in self.metadata.items():
            if key == 'devices_intrinsec_settings':  # Skip non-region entries
                continue

            self.region_dict[key] = Region(oriented_bbox(val["bbox_wgs84"]))

            # Check if satellite imagery exists for this region
            region_view_path = self.root_dir / key / 'satellite' / 'global_view'
            if not region_view_path.exists() or not any(region_view_path.iterdir()):
                self.updated_region.add(key)

    def load_json(self) -> None:
        """Load dataset metadata from JSON file.

        If metadata file exists, loads it and initializes existing regions.
        Otherwise, starts with empty metadata.
        """
        if not self.metadata_json_path.exists():
            self.metadata = {}
        else:
            with open(self.metadata_json_path) as f:
                self.metadata = json.load(f)
            self.init_existing_dataset()

    def save_metadata_json(self) -> None:
        """Save current metadata to JSON file."""
        with open(self.metadata_json_path, "w") as file:
            json.dump(self.metadata, file, indent=4)

    def merge_region(self, regions_to_merge_name):
        list_region_zone = [self.region_dict[region_name].zone for region_name in regions_to_merge_name]
        merge_zone = get_union_of_bboxes(list_region_zone)
        # Create new region acourding to the new bbox
        merge_region_name = self.create_new_region(merge_zone)
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
            self.metadata.pop(former_name)
        return merge_region_name

    def search_corresponding_region(self, zone_candidate: oriented_bbox) -> str:
        """Find or create a region that contains the given zone.

        Searches for existing regions that intersect with the candidate zone.
        If none exist, creates a new region. If multiple exist and merging is enabled,
        merges them into a single region.

        Args:
            zone_candidate: Bounding box to find a region for

        Returns:
            Name of the corresponding region

        Raises:
            ValueError: If multiple regions overlap but merging is disabled
        """
        corresponding_region_candidates = []
        for region_name, region in self.region_dict.items():
            zone_region = region.zone
            zone_intersection = zone_candidate.intersection(zone_region)
            if zone_intersection is not None:
                corresponding_region_candidates.append(region_name)

        if len(corresponding_region_candidates) == 0:
            name_region_in_dict = self.create_new_region(zone_candidate)
            return name_region_in_dict

        elif len(corresponding_region_candidates) == 1:
            corresponding_region_name = corresponding_region_candidates[0]
            corresponding_region = self.region_dict[corresponding_region_name]
            #  If we have a bbox that is not totally inside the Region, we need to update the region
            zone_contained_in = zone_candidate.is_contained_in(corresponding_region.zone)

            if not zone_contained_in:
                # We increase the region by computing the bigger bouding box around the union of bbox
                region_zone_updated = zone_candidate.union(corresponding_region.zone)
                #  Need to update the region
                corresponding_region.update_bbox(region_zone_updated)
                #  Updated it in the region_dict of the Dataset
                self.region_dict[corresponding_region_name] = corresponding_region
                self.metadata[corresponding_region_name]['bbox_wgs84'] = region_zone_updated.get_wgs84_bbox()
                self.updated_region.add(corresponding_region_name)
            return corresponding_region_name
        else:
            if not self.enable_region_merge:
                raise ValueError('Merging Region is not allowed despite having a new area that should require it')
            return self.merge_region(corresponding_region_candidates)

    def create_new_region(self, zone_region: oriented_bbox) -> str:
        """Create a new geographical region in the dataset.

        Args:
            zone_region: Bounding box defining the new region

        Returns:
            Unique name identifier for the new region
        """
        new_region = Region(zone_region.generate_bbox_with_margin(self.margin))
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
        self.metadata[name_region_in_dict] = {'name': name_region,
                                              'bbox_wgs84': zone_region.get_wgs84_bbox(), "zones": {}}
        return name_region_in_dict

    def download_global_tiff_updated_region(self, api_name: str = 'MapTiler') -> None:
        """Download satellite imagery for all regions that have been updated.

        Args:
            api_name: Name of the satellite imagery API service
        """
        for region_name in self.updated_region:
            #  Download the region tiff
            save_path = self.root_dir / region_name / 'satellite' / 'global_view' / f'{api_name}.tiff'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            temp_dir = self.root_dir / region_name / 'satellite' / '.tmp'
            self.region_dict[region_name].download_corresponding_tiff_file(
                save_path, temp_dir, self.api_key, self.margin, self.zoom)

            self.metadata[region_name]['global_view_path'] = str(save_path)
        self.save_metadata_json()

    def retrieve_all_zone_from_region(self, selected_region: str):
        """Get all zone bounding boxes and names from a specific region.

        Args:
            selected_region: Name of the region to query

        Returns:
            Tuple of (list of zone bboxes in WGS84, list of zone names)
        """
        zone_bbox_list = []
        zone_name_list = []
        for zone_name, zone_val in self.metadata[selected_region]['zones'].items():
            zone_bbox_list.append(zone_val["bbox_wgs84"])
            zone_name_list.append(zone_name)
        return zone_bbox_list, zone_name_list

    def extract_candidate_frames(self, video_path: Path, output_root: Path,
                                 output_folder_name: str = 'frames'):
        """Extract candidate frames from drone video based on configuration.

        Args:
            video_path: Path to the input video file
            output_root: Root directory for extracted frames
            output_folder_name: Name of the folder to store frames

        Returns:
            Tuple of (device_config, telemetry_metadata DataFrame)
        """
        if self.verbose >= 1:
            print(f"      Processing: {video_path}")        # Load configurations
        config_data = safe_load(open(self.config_file))
        device_config = load_device_config(
            config_data["device"]["config_path"],
            config_data["device"]["id"]
        )

        # Initialize processor
        telemetry_metadata = filter_video(video_path, self.config_file, output_root / output_folder_name)

        if self.verbose >= 1:
            print(f"      Extracted {len(telemetry_metadata)} frames.")
        return device_config, telemetry_metadata

    def create_zone(self, bbox: oriented_bbox, region_name: str) -> str:
        """Create a new zone within a region.

        Args:
            bbox: Bounding box for the zone
            region_name: Name of the parent region

        Returns:
            Name identifier for the created zone
        """
        # Expand zone with margin for complete coverage
        zone_bbox = bbox.generate_bbox_with_margin(self.margin)
        zone_name = zone_bbox.mercator_name()

        # Initialize zone metadata structure
        self.metadata[region_name]['zones'][zone_name] = {
            'drone': {},
            'satellite': {},
            'bbox_wgs84': zone_bbox.get_wgs84_bbox(),
            'bbox_mercator': zone_bbox.get_mercator_bbox()
        }

        return zone_name

    def process_video(self, video_path: Path, fly_name: str) -> None:
        """Process a drone video through the complete preprocessing pipeline.

        This method performs the following steps:
        1. Extract frames from video based on configuration
        2. Calculate field-of-view (FOV) for each frame
        3. Find or create corresponding geographical region
        4. Prune redundant frames based on IoU and camera angle
        5. Save filtered frames and update metadata

        Args:
            video_path: Path to the drone video file (.MP4)
            fly_name: Identifier for this flight/video session
        """
        if self.verbose >= 1:
            print(f"\n[1/6] Starting Video Processing...")
        tmp_dir = self.root_dir / '.tmp' / video_path.stem
        tmp_dir_folder_name = 'frames'
        device_config, frames_metadata = self.extract_candidate_frames(
            video_path, tmp_dir, output_folder_name=tmp_dir_folder_name)

        if len(frames_metadata) == 0:
            if self.verbose >= 1:
                print(f"No compatible frames detected, skipped {video_path.name}")
            return None

        # --- Step 2: Prepare Camera Geometry ---
        if self.verbose >= 1:
            print(f"\n[2/6] Calculating Field of View (FOV) for all frames...")

        # Get intrinsic matrix (Camera lens properties)
        camera_intrinsics = device_config.cameras.get('EO').intrinsic_settings.K_coefs
        img_width, img_height = device_config.cameras.get('EO').intrinsic_settings.image_size
        #  save device config
        devices_dict = self.metadata.get('devices_intrinsec_settings', {})
        devices_dict[device_config.serial_number] = {
            'K_coefs': device_config.cameras.get('EO').intrinsic_settings.K_coefs,
            'distortion_coefficients': device_config.cameras.get('EO').intrinsic_settings.distortion_coefficients,
            'image_size': device_config.cameras.get('EO').intrinsic_settings.image_size}

        self.metadata['devices_intrinsec_settings'] = devices_dict

        fov_wgs84_list, rotation_matrix_list, global_bouding_box = compute_frames_fov(
            frames_metadata, img_width, img_height, camera_intrinsics)

        # Store both WGS84 and Mercator FOV coordinates
        frames_metadata['fov_wgs84'] = fov_wgs84_list
        # Convert to Mercator in one batch (more efficient)
        fov_mercator_list = [convert_wgs84_to_mercator(coords) for coords in fov_wgs84_list]
        frames_metadata['fov_mercator'] = fov_mercator_list

        # --- Step 3: Download Satellite Imagery ---
        if self.verbose >= 1:
            print(f"\n[3/6] Retrieve corresponding Region")
        # Calculate the North-West and South-East corners for the tile downloader
        drone_zone = oriented_bbox(global_bouding_box)
        corresponding_region_name = self.search_corresponding_region(drone_zone)

        # --- Step 4: Prune Redundant Frames ---
        if self.verbose >= 1:
            print(f"\n[4/6] Pruning Redundant Frames...")

        # Retrieve existing zones to guide frame selection
        guide_coords_list_gps, guide_coords_zone_name = self.retrieve_all_zone_from_region(corresponding_region_name)
        guide_coords_list_mercator = [convert_wgs84_to_mercator(coords) for coords in guide_coords_list_gps]

        guided_matches_output = prune_redundant_areas_with_rotation(
            frames_metadata['fov_mercator'],
            rotation_matrix_list,
            guide_coords_list=guide_coords_list_mercator,
            max_areas_to_keep=len(frames_metadata),  # Initially allow all, let threshold decide
            redundancy_iou_threshold=self.iou_threshold,
            angle_threshold_degrees=self.angle_threshold
        )

        # --- Step 5: Save Filtered Results ---
        if self.verbose >= 1:
            print(f"\n[5/6] Dealing with selected Frames ...")

        working_dir = self.root_dir / corresponding_region_name
        relative_dir = Path('drone') / fly_name
        save_frame_dir = working_dir / relative_dir
        save_frame_dir.mkdir(exist_ok=True, parents=True)

        for frame_idx, zone_idx in guided_matches_output.items():
            frame_info = frames_metadata.iloc[frame_idx]
            frame_info_dict = frame_info.to_dict()
            frame_filename = f"{video_path.stem}_{frame_info.name}.jpg"
            frame_zone = oriented_bbox(frame_info_dict['fov_wgs84'])

            # Assign frame to existing zone or create new one
            if zone_idx is None:
                zone_name = self.create_zone(frame_zone, corresponding_region_name)
            else:
                zone_name = guide_coords_zone_name[zone_idx]

            # Copy frame file to destination
            source_path = tmp_dir / tmp_dir_folder_name / frame_filename
            dest_path = save_frame_dir / frame_filename
            if source_path.exists():
                shutil.copy(source_path, dest_path)
            else:
                if self.verbose >= 1:
                    print(f"      Warning: Source file missing {source_path}")

            # Update metadata with frame information
            frame_id = dest_path.stem
            frame_metadata = {
                'image_path': str(relative_dir / frame_filename),
                'fly_id': fly_name,
                **frame_info_dict
            }
            self.metadata[corresponding_region_name]['zones'][zone_name]['drone'][frame_id] = frame_metadata

    def download_tiles_corresponding_to_drone(self) -> None:
        """Extract satellite tiles for each zone from global satellite imagery.

        Iterates through all regions and zones, extracting the relevant portion
        of the global satellite view for each zone's bounding box.
        """
        for region_name, content in self.metadata.items():
            if region_name == 'devices_intrinsec_settings':  # Skip non-region entries
                continue
            global_view_folder = self.root_dir / region_name / 'satellite' / 'global_view'
            zones_dict = content['zones']
            for global_view in global_view_folder.iterdir():
                if global_view.suffix != '.tiff':
                    continue
                for zone_id, zone_content in zones_dict.items():
                    zone_mercator = zone_content['bbox_mercator']
                    relative_path = Path('satellite') / 'local_tiles' / zone_id / global_view.name
                    save_tiff_file = self.root_dir / region_name / relative_path
                    save_tiff_file.parent.mkdir(exist_ok=True, parents=True)
                    self.metadata[region_name]["zones"][zone_id]["satellite"][global_view.stem] = str(relative_path)
                    if save_tiff_file.exists():
                        continue
                    tile_result = get_best_tile_for_fov(global_view, zone_mercator, verbose=self.verbose)
                    if tile_result:
                        save_tile_to_disk(tile_result, str(save_tiff_file), verbose=self.verbose)
        self.save_metadata_json()

    def run_extraction(self) -> None:
        """Execute the complete extraction pipeline for all unprocessed videos.

        Processes each video in the video directory that hasn't been extracted yet,
        then downloads all necessary satellite imagery.
        """
        already_extracted_flight = list(find_values_gen(self.metadata, "fly_id"))
        for fly_folder in self.video_dir.iterdir():
            if fly_folder.name in already_extracted_flight:
                continue
            for video in fly_folder.iterdir():
                if video.suffix.lower() != '.mp4':
                    continue
                if video.stem.split('_')[-1] != 'V':
                    continue
                self.process_video(video, fly_folder.name)
        self.save_metadata_json()
        self.download_after_extraction()

    def download_after_extraction(self) -> None:
        """Download all satellite imagery after video extraction is complete.

        First downloads global satellite views for updated regions,
        then extracts local tiles for each zone.
        """
        self.download_global_tiff_updated_region()
        self.download_tiles_corresponding_to_drone()


if __name__ == '__main__':
    root_dir = 'data/preprocessing_extraction_sans_crotalinea'
    video_dir = '/home/sebastienmorel/Documents/Data/raw-flights/raw'
    config_file = 'config/preprocessing/raw_preprocessing_config_refacto.yaml'
    margin = 0.1
    zoom = 17
    api_key = "SZ5Q6ilGzFm9Wge4GYp8"

    # verbose levels:
    # 0 = silent (no output)
    # 1 = normal (default, shows progress)
    # 2 = debug (shows additional debug info)
    test_dataset = Dataset(root_dir, video_dir, config_file, margin, zoom, api_key, verbose=0)
    test_dataset.run_extraction()
