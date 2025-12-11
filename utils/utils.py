from scipy.spatial.transform import Rotation as R_scipy
from rasterio.warp import transform

from pathlib import Path
import shutil


def convert_mercator_to_wgs84(fov_web_mercator, api_order=False):
    """
    Converts a list of EPSG:3857 (Web Mercator) points to EPSG:4326 (Lat/Lon)
    using rasterio.

    :param fov_web_mercator: List of tuples [(x, y), (x, y), ...]
    :return: List of tuples [(lat, lon), (lat, lon), ...]
    """
    if not fov_web_mercator:
        return []

    # 1. Unzip the list of tuples into two separate lists (Xs and Ys)
    # Rasterio expects: ([x1, x2], [y1, y2])
    xs = [pt[0] for pt in fov_web_mercator]
    ys = [pt[1] for pt in fov_web_mercator]

    # 2. Perform the transformation
    # src_crs='EPSG:3857' (Web Mercator)
    # dst_crs='EPSG:4326' (WGS84 Lat/Lon)
    lons, lats = transform(src_crs='EPSG:3857', dst_crs='EPSG:4326', xs=xs, ys=ys)

    # 3. Zip them back together into (lat, lon) tuples
    if api_order:
        # For api, convention is usually lon, lat
        return list(zip(lons, lats))
    else:
        return list(zip(lats, lons))


def convert_wgs84_to_mercator(wgs84_points):
    """
    Converts a list of EPSG:4326 (Lat, Lon) points to EPSG:3857 (Web Mercator X, Y).
    Optional: can be a list of (Lat, Lon, alt) points that will give (X, Y, H)

    :param wgs84_points: List of tuples [(lat, lon), (lat, lon), ...]
                         Note: Ensure input is (Latitude, Longitude), not (Lon, Lat).
    :return: List of tuples [(x, y), (x, y), ...]
    """
    if not wgs84_points:
        return []
    if len(wgs84_points[0]) == 3:
        alts = [pt[2] for pt in wgs84_points]
    else:
        alts = None
    # 1. Unzip the list into Longitudes and Latitudes
    lons = [pt[1] for pt in wgs84_points]
    lats = [pt[0] for pt in wgs84_points]

    # 2. Perform the transformation
    # src_crs='EPSG:4326' (WGS84) -> dst_crs='EPSG:3857' (Web Mercator)
    xs, ys = transform(src_crs='EPSG:4326', dst_crs='EPSG:3857', xs=lons, ys=lats)

    # 3. Zip them back together into (x, y) tuples
    if alts is None:
        return list(zip(xs, ys))
    else:
        return list(zip(xs, ys, alts))


def get_euler_angles_scipy(R_matrix):
    """
    Returns Yaw, Pitch, Roll in degrees.
    Convention: 'xyz' (extrinsic).
    """
    # Create a rotation object from the matrix
    r = R_scipy.from_matrix(R_matrix)

    # Convert to Euler angles
    # 'xyz' is the standard convention for cameras
    yaw, pitch, roll = r.as_euler('xyz', degrees=True)

    return yaw, pitch, roll


def transfer_skip_existing_names(source_folder, target_folder):
    """
    Recursively moves files from source to target using pathlib.
    Skips files if the name already exists in the destination.
    """
    # Convert strings to Path objects
    source = Path(source_folder)
    target = Path(target_folder)

    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    files_skipped = 0

    print(f"--- Transferring: {source} -> {target} ---")

    # rglob('*') recursively iterates over all files and folders
    for src_file in source.rglob('*'):
        if src_file.is_file():
            # Calculate the relative path (e.g. 'subfolder/image.jpg')
            relative_path = src_file.relative_to(source)

            # Construct the full destination path
            # (pathlib allows using '/' operator to join paths)
            dest_file = target / relative_path

            # --- THE CHECK ---
            if dest_file.exists():
                print(f"[SKIP] {dest_file.name}")
                files_skipped += 1
            else:
                # Ensure the specific sub-folder exists before copying
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    src_file.rename(dest_file)
                    print(f"[MOVE] {dest_file.name}")
                    files_copied += 1
                except Exception as e:
                    print(f"[ERROR] {src_file.name}: {e}")

    print(f"\n--- Done ---")
    print(f"Copied:  {files_copied}")
    print(f"Skipped: {files_skipped}")


def delete_folder(folder_path):
    """
    Deletes a folder and all its contents.
    """
    target = Path(folder_path)

    # 1. Check if the folder actually exists
    if not target.exists():
        print(f"[Error] The folder '{folder_path}' does not exist.")
        return

    # 2. Check if it is actually a directory (not a file)
    if not target.is_dir():
        print(f"[Error] '{folder_path}' is a file, not a folder.")
        return

    # 3. Attempt to delete
    try:
        # shutil.rmtree is the function that handles recursive deletion
        shutil.rmtree(target)
        print(f"[Success] Deleted folder: {folder_path}")
    except OSError as e:
        print(f"[Error] Failed to delete {folder_path}. Reason: {e}")
