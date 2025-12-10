from scipy.spatial.transform import Rotation as R_scipy
from rasterio.warp import transform
import numpy as np
from pathlib import Path
import shutil


def get_union_of_bboxes(bbox_list):
    """
    Computes the smallest bounding box that contains all boxes in the list.

    :param bbox_list: List of tuples [ (nw, se), (nw, se), ... ]
                      where nw=(lat_max, lon_min) and se=(lat_min, lon_max)
    :return: Tuple (nw_union, se_union) or None if list is empty
    """
    if not bbox_list:
        return None

    # Initialize extremes with the first box
    first_nw, first_se = bbox_list[0]

    # Current extremes
    union_lat_max = first_nw[0]  # North
    union_lon_min = first_nw[1]  # West
    union_lat_min = first_se[0]  # South
    union_lon_max = first_se[1]  # East

    # Iterate through the remaining boxes
    for i in range(1, len(bbox_list)):
        nw, se = bbox_list[i]
        curr_lat_max, curr_lon_min = nw
        curr_lat_min, curr_lon_max = se

        # Expand boundaries if current box goes further out
        if curr_lat_max > union_lat_max:
            union_lat_max = curr_lat_max  # More North
        if curr_lon_min < union_lon_min:
            union_lon_min = curr_lon_min  # More West
        if curr_lat_min < union_lat_min:
            union_lat_min = curr_lat_min  # More South
        if curr_lon_max > union_lon_max:
            union_lon_max = curr_lon_max  # More East

    return (union_lat_max, union_lon_min), (union_lat_min, union_lon_max)


def is_bbox_inside(inner_box, outer_box):
    """
    Checks if inner_box is completely contained within outer_box.

    Parameters:
        inner_box: Tuple (nw, se) where nw=(lat_max, lon_min), se=(lat_min, lon_max)
        outer_box: Tuple (nw, se) where nw=(lat_max, lon_min), se=(lat_min, lon_max)
    """
    # Unpack coordinates for clarity
    # Box = [ (lat_max, lon_min), (lat_min, lon_max) ]
    in_nw, in_se = inner_box
    out_nw, out_se = outer_box

    # 1. Check Latitude (North/South)
    # Inner North must be <= Outer North
    cond_north = in_nw[0] <= out_nw[0]
    # Inner South must be >= Outer South
    cond_south = in_se[0] >= out_se[0]

    # 2. Check Longitude (West/East)
    # Inner West must be >= Outer West
    cond_west = in_nw[1] >= out_nw[1]
    # Inner East must be <= Outer East
    cond_east = in_se[1] <= out_se[1]

    return cond_north and cond_south and cond_west and cond_east


def add_percentage_margin_to_bbox(nw, se, margin_percentage):
    """
    Expands a (lat, lon) bounding box by a percentage of its current size.

    :param nw: Tuple (lat_max, lon_min)
    :param se: Tuple (lat_min, lon_max)
    :param margin_percentage: Float (e.g., 0.10 for 10% margin)
    :return: (new_nw, new_se)
    """
    lat_max, lon_min = nw
    lat_min, lon_max = se

    # 1. Calculate current dimensions (in degrees)
    height_deg = lat_max - lat_min
    width_deg = lon_max - lon_min

    # 2. Calculate the "padding" amount
    # (margin * dimension)
    lat_padding = height_deg * margin_percentage
    lon_padding = width_deg * margin_percentage

    # 3. Apply padding
    # Expand North (+) and South (-)
    new_lat_max = lat_max + lat_padding
    new_lat_min = lat_min - lat_padding

    # Expand East (+) and West (-)
    new_lon_max = lon_max + lon_padding
    new_lon_min = lon_min - lon_padding

    return (new_lat_max, new_lon_min), (new_lat_min, new_lon_max)


def compute_intersection(box1, box2):
    """
    Computes the intersection bounding box of two bounding boxes.

    Parameters:
        box1: (N, 4) or (4,) numpy array [x_min, y_min, x_max, y_max]
        box2: (N, 4) or (4,) numpy array [x_min, y_min, x_max, y_max]

    Returns:
        intersection_box: (N, 4) or (4,) numpy array [x_min, y_min, x_max, y_max]
                          Returns zeros if there is no intersection.
    """
    # Ensure inputs are numpy arrays
    b1 = np.array(box1)
    b2 = np.array(box2)

    # 1. Calculate the intersection coordinates
    # Intersection Top-Left is the MAX of the two Top-Lefts
    inter_min = np.maximum(b1[..., :2], b2[..., :2])

    # Intersection Bottom-Right is the MIN of the two Bottom-Rights
    inter_max = np.minimum(b1[..., 2:], b2[..., 2:])

    # 2. Compute the dimensions of the intersection box
    # If max < min, there is no overlap, so we clip the dimension to 0
    dims = np.maximum(inter_max - inter_min, 0)

    # 3. Reconstruct the box [x_min, y_min, x_max, y_max]
    # If dimensions are 0 (no overlap), we often want the box to be [0,0,0,0]
    # or just the valid slice with 0 area.
    # Here we check if area is 0 to zero out the coordinates for cleanliness.

    # Create the intersection box candidate
    inter_box = np.concatenate([inter_min, inter_max], axis=-1)

    # Mask out invalid boxes where x_min > x_max or y_min > y_max (width or height is 0)
    # Check if width or height is 0
    valid_mask = np.all(dims > 0, axis=-1)

    # Apply mask: where invalid, return 0
    if inter_box.ndim == 1:
        return inter_box if valid_mask else np.zeros_like(inter_box)
    else:
        inter_box[~valid_mask] = 0
        return inter_box


def compute_enclosing_box(box1, box2):
    """
    Computes the Smallest Enclosing Bounding Box (Merge).
    """
    b1 = np.array(box1)
    b2 = np.array(box2)

    # Top-Left is the MIN of the two Top-Lefts
    enc_min = np.minimum(b1[..., :2], b2[..., :2])

    # Bottom-Right is the MAX of the two Bottom-Rights
    enc_max = np.maximum(b1[..., 2:], b2[..., 2:])

    # Concatenate
    return np.concatenate([enc_min, enc_max], axis=-1)


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


def convert_wgs84_to_web_mercator(wgs84_points):
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
    Recursively copies files from source to target using pathlib.
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
                    shutil.copy2(src_file, dest_file)
                    print(f"[COPY] {dest_file.name}")
                    files_copied += 1
                except Exception as e:
                    print(f"[ERROR] {src_file.name}: {e}")

    print(f"\n--- Done ---")
    print(f"Copied:  {files_copied}")
    print(f"Skipped: {files_skipped}")
