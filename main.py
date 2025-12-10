from pathlib import Path
from yaml import safe_load
import numpy as np
import shutil


# Custom module imports
from crotalinae_pg.data_preprocessing.raw.video_processor import VideoProcessor
from crotalinae_pg.data_preprocessing.raw.utils.devices import load_device_config
from retrieve_position_from_UAV_view import process as calculate_fov_coords, get_R as get_rotation_matrix, get_calibration_matrix
from retreieve_satellite_image_depending_on_coord import download_tiles, get_bounding_box, merge_tiles_to_geotiff
from pruning_tile import prune_redundant_areas_with_rotation
from extract_satellite_tile_from_drone_view import get_best_tile_for_fov, save_tile_to_disk
from pnp import get_camera_position_robust
from utils.QGIS_generation_files import generate_multi_camera_geojson, generate_position_comparison_geojson
from utils.utils import convert_wgs84_to_web_mercator, convert_mercator_to_wgs84


def extract_candidate_frames(video_path, output_root, config_path, frames_folder_name):
    """
    Extracts frames from the video based on the delta_frames configuration.
    """
    print(f"\n[1/6] Starting Video Processing...")
    print(f"      Input: {video_path}")

    # Load configurations
    config_data = safe_load(open(config_path))
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
    video_processor.process(Path(video_path), frames_folder_name)

    # Retrieve the dataframe containing telemetry (Lat, Lon, Yaw, Pitch, etc.)
    telemetry_metadata = video_processor.frame_saver.metadata

    print(f"      Extracted {len(telemetry_metadata)} frames.")
    return device_config, telemetry_metadata


def main(video_path, output_root, config_path, api_key, zoom_level, iou_threshold=0.5, angle_threshold=15, frames_folder_name='frames_candidates', verbose=0):

    # --- Step 1: Extract Frames from Video ---
    device_config, metadata_df = extract_candidate_frames(video_path, output_root, config_path, frames_folder_name)

    # Define working directory for this specific video
    working_dir = Path(output_root) / Path(video_path).stem

    # --- Step 2: Prepare Camera Geometry ---
    print(f"\n[2/6] Calculating Field of View (FOV) for all frames...")

    # Get intrinsic matrix (Camera lens properties)
    camera_intrinsics = device_config.cameras.get('EO').intrinsic_settings.K_coefs
    img_width, img_height = device_config.cameras.get('EO').intrinsic_settings.image_size

    # Define the 4 corners of the image in pixel coordinates (Homogeneous coords)
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right : Need to have W -1.
    image_corners_homogeneous = np.array([
        [0, 0, 1],
        [img_width - 1, 0, 1],
        [0, img_height - 1, 1],
        [img_width - 1, img_height - 1, 1]
    ])

    fov_wgs84_list = []      # Will store 4 corner points (Lat/Lon) for each frame
    rotation_matrix_list = []  # Will store camera rotation matrix for each frame

    # Track min/max coordinates to know which satellite area to download later
    global_max_lat, global_min_lat = -np.inf, np.inf
    global_max_lon, global_min_lon = -np.inf, np.inf

    # Iterate through every extracted frame
    for index, row in metadata_df.iterrows():
        # Extract telemetry
        lat, lon = row["Latitude"], row['Longitude']
        alt, rel_alt = row['Absolute Altitude'], row['Relative Altitude']
        pitch, yaw, roll = row['Gimbal Pitch'], row['Gimbal Yaw'], row['Gimbal Roll']

        # 1. Project image corners to the ground (WGS84 Coordinates)
        fov_coords = calculate_fov_coords(
            lat, lon, alt, rel_alt,
            pitch, yaw, roll,
            camera_intrinsics, image_corners_homogeneous
        )
        fov_wgs84_list.append(fov_coords)

        # 2. Calculate Rotation Matrix (for viewing angle pruning)
        r_mat = get_rotation_matrix(pitch, yaw, roll)
        rotation_matrix_list.append(r_mat)

        # 3. Update Global Bounding Box
        for point in fov_coords:
            p_lat, p_lon = point[:2]
            global_max_lat = max(p_lat, global_max_lat)
            global_min_lat = min(p_lat, global_min_lat)
            global_max_lon = max(p_lon, global_max_lon)
            global_min_lon = min(p_lon, global_min_lon)

    # Store results back in dataframe
    metadata_df['fov_wgs84'] = fov_wgs84_list

    # --- Step 3: Download Satellite Imagery ---
    print(f"\n[3/6] Downloading Satellite Imagery...")

    # Calculate the North-West and South-East corners for the tile downloader
    nw_corner, se_corner = get_bounding_box(
        (global_min_lat, global_min_lon),
        (global_max_lat, global_max_lon)
    )

    satellite_download_dir = working_dir / f"Tiles_z{zoom_level}"
    tiles_storage_dir = satellite_download_dir / 'zone_tiles'
    satellite_download_dir.mkdir(exist_ok=True, parents=True)

    print(f"      Region: {nw_corner} to {se_corner}")
    download_tiles(nw_corner, se_corner, api_key, zoom=zoom_level, output_dir=tiles_storage_dir)

    # Merge individual small tiles into one big GeoTIFF
    merged_tiff_path = satellite_download_dir / "Flight_zone.tiff"
    print(f"      Merging tiles to: {merged_tiff_path}")
    merge_tiles_to_geotiff(tiles_storage_dir, merged_tiff_path, zoom=zoom_level)

    # --- Step 4: Prune Redundant Frames ---
    print(f"\n[4/6] Pruning Redundant Frames...")

    # Convert FOV coordinates to Web Mercator (Meters) for accurate area calculation
    metadata_df['fov_mercator'] = [convert_wgs84_to_web_mercator(coords) for coords in metadata_df['fov_wgs84']]

    # Save updated metadata before pruning
    metadata_df.to_csv(working_dir / frames_folder_name / "metadata_fov.csv")

    # Run the Smart Filter (IoU + Rotation check)
    kept_indices = prune_redundant_areas_with_rotation(
        metadata_df['fov_mercator'].tolist(),
        rotation_matrix_list,
        max_areas_to_keep=len(metadata_df),  # Initially allow all, let threshold decide
        iou_threshold=iou_threshold,
        angle_threshold_degrees=angle_threshold
    )

    # Filter the dataframe
    filtered_metadata = metadata_df.iloc[kept_indices]

    print(f"      Original Frames: {len(metadata_df)}")
    print(f"      Kept Frames:     {len(filtered_metadata)}")

    # --- Step 5: Save Filtered Results ---
    print(f"\n[5/6] Copying Selected Frames...")

    prune_output_dir = working_dir / f'Pruned_Frames_ioU{iou_threshold}_angle{angle_threshold}'
    prune_output_dir.mkdir(exist_ok=True, parents=True)

    # Save the cleaned metadata
    filtered_metadata.to_csv(prune_output_dir / 'metadata.csv')

    # Copy the actual image files
    for frame_filename in filtered_metadata.index.values:
        source_path = working_dir / frames_folder_name / frame_filename
        dest_path = prune_output_dir / frame_filename

        if source_path.exists():
            shutil.copy(source_path, dest_path)
        else:
            print(f"      Warning: Source file missing {source_path}")

    # --- Step 6: Extract Matching Satellite Tiles ---
    print(f"\n[6/6] Extracting Matched Satellite Tiles...")

    extracted_tiles_dir = satellite_download_dir / 'extracted_tiles'
    extracted_tiles_dir.mkdir(exist_ok=True, parents=True)

    # Iterate over the kept frames
    count = 0
    for fov_coords, media_name in zip(filtered_metadata['fov_mercator'], filtered_metadata.index.values):
        media_stem = Path(media_name).stem  # Remove extension (e.g., .jpg)

        # Crop the big Tiff to match this specific frame's view
        tile_result = get_best_tile_for_fov(merged_tiff_path, fov_coords)

        if tile_result:
            save_tile_to_disk(tile_result, str(extracted_tiles_dir / f'{media_stem}_sat.tiff'))
            count += 1

    print(f"      Successfully saved {count} satellite crops.")

    if verbose > 0:
        # --- Step Optional: Compute retrieval of camera position and QGIS files ---
        print(f"\n[Optional Step] Compute QGIS files and camera position estimation...")
        #  Need calibration Matrix
        calibration_matrix = get_calibration_matrix(camera_intrinsics)
        #  Need list of dict for the QGIS files
        list_dict_pos_estimation = []
        list_dict_pos_fov = []
        # Saving dir
        qgis_saving_dir = working_dir / 'QGIS_files'
        qgis_saving_dir.mkdir(exist_ok=True, parents=True)
        #  Compute also the mean error of positional estimation of the camera as validation of the pipeline
        error_estimation_norm = []

        for frame_idx, lat, lon, fov_mercator in filtered_metadata[['FrameNumber', 'Latitude', 'Longitude', 'fov_mercator']].values:
            # Retrieve the position of the camera with PnP algorithm
            pose_estimation = get_camera_position_robust(image_corners_homogeneous[:, :2],
                                                         np.array(fov_mercator), calibration_matrix)
            #  Convert the real position to mercator for computing error of position
            pos_mercator = convert_wgs84_to_web_mercator([[lat, lon]])[0]
            error_estimation_norm.append(np.linalg.norm(np.array(pose_estimation)[:2] - np.array(pos_mercator)))
            #  Convert the estimated position to gps for QGIS functions
            pose_estimation_gps = convert_mercator_to_wgs84([pose_estimation])[0]
            #  Update the list of dicts
            list_dict_pos_estimation.append({"id": frame_idx, "real": (lat, lon),
                                            "estimated": (pose_estimation_gps[0], pose_estimation_gps[1])})
            list_dict_pos_fov.append({"id": frame_idx, "pos": (lat, lon), "fov": fov_mercator})

        generate_position_comparison_geojson(list_dict_pos_estimation,
                                             output_file=qgis_saving_dir / "Estimation_camera_position.geojson")
        generate_multi_camera_geojson(list_dict_pos_fov,
                                      output_file=qgis_saving_dir / "fov_position.geojson")
        mean_error_estimation = np.mean(error_estimation_norm)
        print(f"\nMean Error of retrieving the camera is {mean_error_estimation}m")

        print(f"\nDone! Results in: {working_dir}")


if __name__ == '__main__':
    # Configuration
    VIDEO_PATH = 'data/raw_fights/DJI_202509271440_017/DJI_20250927144136_0001_V.MP4'
    OUTPUT_ROOT = 'data/preprocessing'
    CONFIG_FILE = "config/preprocessing/raw_preprocessing_config.yaml"
    API_KEY = "SZ5Q6ilGzFm9Wge4GYp8"  # Be careful not to commit real keys to git!

    # Parameters
    ZOOM_LEVEL = 18
    IOU_THRESHOLD = 0.5
    ANGLE_THRESHOLD = 15
    VERBOSE = 1

    main(
        VIDEO_PATH,
        OUTPUT_ROOT,
        CONFIG_FILE,
        API_KEY,
        ZOOM_LEVEL,
        iou_threshold=IOU_THRESHOLD,
        angle_threshold=ANGLE_THRESHOLD,
        verbose=VERBOSE
    )
