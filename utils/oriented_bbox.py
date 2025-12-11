from typing import List, Tuple, Optional
import numpy as np
from utils.utils import convert_wgs84_to_mercator


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

    def __init__(self, bbox_wgs_84: List[Tuple[float, float]]):
        """
        Initializes the bounding box from a list of coordinates.

        Args:
            bbox_wgs_84 (List[Tuple[float, float]]): A list of (Lat, Lon) points defining the region.
                This can be just 2 points (NW, SE) or a polygon of points.
            margin (float, optional): A percentage to expand the bounding box on all sides.
                Defaults to 0. Example: 0.1 expands width and height by 10%.
        """
        self.nw, self.se = get_bounding_box(bbox_wgs_84)
        self.nw_mercator, self.se_mercator = convert_wgs84_to_mercator([self.nw, self.se])

    def generate_bbox_with_margin(self, margin=0) -> None:
        """
        Internal method to expand the bounding box dimensions based on `self.margin`.
        Modifies `self.nw` and `self.se` in-place.
        """
        lat_max, lon_min = self.nw
        lat_min, lon_max = self.se
        height = lat_max - lat_min
        width = lon_max - lon_min

        lat_pad = height * margin
        lon_pad = width * margin

        nw = (lat_max + lat_pad, lon_min - lon_pad)
        se = (lat_min - lat_pad, lon_max + lon_pad)

        return oriented_bbox([nw, se])

    @property
    def center(self) -> Tuple[float, float]:
        """
        Calculates the geometric center (centroid) of the bounding box.

        Returns:
            Tuple[float, float]: The (Latitude, Longitude) of the center point.
        """
        # Average of Top and Bottom Latitudes
        center_lat = (self.nw[0] + self.se[0]) / 2.0

        # Average of Left and Right Longitudes
        center_lon = (self.nw[1] + self.se[1]) / 2.0

        return center_lat, center_lon

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

    def mercator_name(self):
        # Cast to integer (truncates decimals) and format string
        # Using int() removes the decimal point entirely (e.g. 123.45 -> 123)
        x1, y1 = self.nw_mercator
        x2, y2 = self.se_mercator
        return f"{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}"

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
