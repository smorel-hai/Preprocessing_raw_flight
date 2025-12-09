from scipy.spatial.transform import Rotation as R_scipy
from rasterio.warp import transform


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
