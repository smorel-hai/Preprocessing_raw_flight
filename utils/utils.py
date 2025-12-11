from typing import List, Tuple, Optional
import math
from scipy.spatial.transform import Rotation as R_scipy
from rasterio.warp import transform
import numpy as np
from pathlib import Path
import shutil
from typing import List


def get_bounding_box(coordinate_list: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes the North-West and South-East corners of a North-aligned bounding box
    that encloses the given set of coordinates.

    Args:
        coordinate_list (List[Tuple[float, float]]): A list of (Latitude, Longitude) points.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]: A tuple containing:
            - NW corner (lat_max, lon_min)
            - SE corner (lat_min, lon_max)
    """
    coordinate_array = np.array(coordinate_list)
    lats = coordinate_array[:, 0]
    lons = coordinate_array[:, 1]

    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_lon = np.max(lons)
    min_lon = np.min(lons)

    nw_corner = (max_lat, min_lon)
    se_corner = (min_lat, max_lon)
    return nw_corner, se_corner


class oriented_bbox:
    """
    A class representing a North-aligned Bounding Box defined by geographical coordinates (WGS84).
    Supports margin expansion, geometric operations (intersection, union), and coordinate projection.

    Attributes:
        nw (Tuple[float, float]): The North-West corner (Latitude, Longitude).
        se (Tuple[float, float]): The South-East corner (Latitude, Longitude).
        margin (float): The percentage margin applied to expand the box (e.g., 0.1 for 10%).
    """

    def __init__(self, bbox_wgs_84: List[Tuple[float, float]], margin: float = 0):
        """
        Initializes the bounding box from a list of coordinates.

        Args:
            bbox_wgs_84 (List[Tuple[float, float]]): A list of (Lat, Lon) points defining the region.
                This can be just 2 points (NW, SE) or a polygon of points.
            margin (float, optional): A percentage to expand the bounding box on all sides.
                Defaults to 0. Example: 0.1 expands width and height by 10%.
        """
        self.nw, self.se = get_bounding_box(bbox_wgs_84)
        self.margin = margin

        if self.margin != 0:
            self._apply_margin()

    def _apply_margin(self) -> None:
        """
        Internal method to expand the bounding box dimensions based on `self.margin`.
        Modifies `self.nw` and `self.se` in-place.
        """
        lat_max, lon_min = self.nw
        lat_min, lon_max = self.se
        height = lat_max - lat_min
        width = lon_max - lon_min

        lat_pad = height * self.margin
        lon_pad = width * self.margin

        self.nw = (lat_max + lat_pad, lon_min - lon_pad)
        self.se = (lat_min - lat_pad, lon_max + lon_pad)

    def get_wgs84_bbox(self) -> List[Tuple[float, float]]:
        """
        Retrieves the four corner coordinates of the bounding box in WGS84.

        Returns:
            List[Tuple[float, float]]: A list of 4 points in clockwise order:
            [Top-Left (NW), Top-Right (NE), Bottom-Right (SE), Bottom-Left (SW)].
        """
        top_left = self.nw
        top_right = (self.nw[0], self.se[1])    # Lat_Max, Lon_Max
        bottom_right = self.se
        bottom_left = (self.se[0], self.nw[1])   # Lat_Min, Lon_Min

        return [top_left, top_right, bottom_right, bottom_left]

    def get_mercator_bbox(self) -> List[List[float]]:
        """
        Retrieves the four corner coordinates projected into Web Mercator (EPSG:3857).

        Returns:
            List[List[float]]: A list of 4 points [x, y] in meters, corresponding to the
            WGS84 corners.
        """
        wgs84_corners = self.get_wgs84_bbox()
        return convert_wgs84_to_mercator(wgs84_corners)

    def intersection(self, other: 'oriented_bbox') -> Optional['oriented_bbox']:
        """
        Computes the intersection (overlapping region) between this bbox and another.

        Args:
            other (oriented_bbox): The other bounding box to compare against.

        Returns:
            Optional[oriented_bbox]: A new oriented_bbox representing the intersection,
            or None if the boxes do not overlap.
        """
        s_lat_max, s_lon_min = self.nw
        s_lat_min, s_lon_max = self.se
        o_lat_max, o_lon_min = other.nw
        o_lat_min, o_lon_max = other.se

        int_lat_max = min(s_lat_max, o_lat_max)
        int_lat_min = max(s_lat_min, o_lat_min)
        int_lon_min = max(s_lon_min, o_lon_min)
        int_lon_max = min(s_lon_max, o_lon_max)

        # Check if valid (North > South AND East > West)
        if int_lat_max < int_lat_min or int_lon_max < int_lon_min:
            return None

        new_coords = [(int_lat_max, int_lon_min), (int_lat_min, int_lon_max)]
        return oriented_bbox(new_coords, margin=0)

    def is_contained_in(self, other: 'oriented_bbox') -> bool:
        """
        Determines if this bounding box is strictly contained within another.

        Args:
            other (oriented_bbox): The bounding box to check against (the potential container).

        Returns:
            bool: True if `self` is completely inside `other`, False otherwise.
        """
        s_lat_max, s_lon_min = self.nw
        s_lat_min, s_lon_max = self.se
        o_lat_max, o_lon_min = other.nw
        o_lat_min, o_lon_max = other.se

        return (s_lat_max <= o_lat_max and
                s_lat_min >= o_lat_min and
                s_lon_min >= o_lon_min and
                s_lon_max <= o_lon_max)

    def union(self, other_bbox: 'oriented_bbox') -> 'oriented_bbox':
        """
        Computes the union (Smallest Enclosing Bounding Box) of this box and another.

        Args:
            other_bbox (oriented_bbox): The other bounding box to merge with.

        Returns:
            oriented_bbox: A new bounding box that minimally encloses both original boxes.
        """
        s_lat_max, s_lon_min = self.nw
        s_lat_min, s_lon_max = self.se
        o_lat_max, o_lon_min = other_bbox.nw
        o_lat_min, o_lon_max = other_bbox.se

        uni_lat_max = max(s_lat_max, o_lat_max)
        uni_lat_min = min(s_lat_min, o_lat_min)
        uni_lon_min = min(s_lon_min, o_lon_min)
        uni_lon_max = max(s_lon_max, o_lon_max)

        new_coords = [(uni_lat_max, uni_lon_min), (uni_lat_min, uni_lon_max)]
        return oriented_bbox(new_coords, margin=0)

    def __repr__(self):
        return f"oriented_bbox(NW={self.nw}, SE={self.se})"


def get_union_of_bboxes(bbox_list: List[oriented_bbox]) -> Optional[oriented_bbox]:
    """
    Computes the smallest bounding box that contains all bounding boxes in a list.

    Args:
        bbox_list (List[oriented_bbox]): A list of oriented_bbox objects.

    Returns:
        Optional[oriented_bbox]: A single bounding box enclosing all input boxes,
        or None if the input list is empty.
    """
    if not bbox_list:
        return None

    # Initialize union with the first box and iteratively merge
    union_bbox = bbox_list[0]
    for bbox in bbox_list[1:]:
        union_bbox = union_bbox.union(bbox)

    return union_bbox


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
