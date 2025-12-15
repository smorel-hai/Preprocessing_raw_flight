"""UAV camera geometry and field-of-view calculation module.

This module computes the ground footprint (Field of View) of drone camera frames
using camera intrinsics, telemetry data, and coordinate transformations.

Key functions:
- get_rotation_matrix: Convert Euler angles to rotation matrix
- project_fov_to_ground: Project camera pixels to ground coordinates
- compute_frames_fov: Batch process multiple frames
"""

import numpy as np


def get_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Compute the rotation matrix from Body Frame to NED (North-East-Down) Frame.

    Applies rotations in order: Yaw -> Pitch -> Roll (standard aerospace convention).

    Args:
        pitch: Pitch angle in degrees (nose up/down)
        yaw: Yaw angle in degrees (heading, 0=North, clockwise positive)
        roll: Roll angle in degrees (wing tilt)

    Returns:
        3x3 rotation matrix for transforming body frame to NED frame
    """
    roll_rad = np.deg2rad(roll)
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)

    # Rotation around X-axis (Roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad), np.cos(roll_rad)]])

    # Rotation around Y-axis (Pitch)
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    # Rotation around Z-axis (Yaw)
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                   [0, 0, 1]])

    # Transformation from Body to NED = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def get_meridional_radius(phi: float, a: float = 6378137.0, e2: float = 0.0066943799901413165) -> float:
    """Calculate Meridional Radius of Curvature. (Rn in confluence).

    Args:
        phi: Latitude in radians
        a: Earth semi-major axis (WGS84 default)
        e2: Earth eccentricity squared (WGS84 default)

    Returns:
        Meridional radius in meters
    """
    return (a * (1 - e2) / np.power((1 - e2 * np.sin(phi)**2), 3/2))


def get_prime_vertical_radius(phi: float, a: float = 6378137.0, e2: float = 0.0066943799901413165) -> float:
    """Calculate Prime Vertical Radius of Curvature (Re in confluence).

    Args:
        phi: Latitude in radians
        a: Earth semi-major axis (WGS84 default)
        e2: Earth eccentricity squared (WGS84 default)

    Returns:
        Prime vertical radius in meters
    """
    return a / np.sqrt(1 - e2 * np.sin(phi)**2)


def get_lat_lon_alt_from_NED_dep(x: float, y: float, z: float,
                                 phi0_deg: float, lam0_deg: float, h0: float) -> tuple:
    """Convert NED (North-East-Down) offsets to geodetic coordinates.

    Args:
        x: North offset in meters
        y: East offset in meters  
        z: Down offset in meters (positive down)
        phi0_deg: Reference latitude in degrees
        lam0_deg: Reference longitude in degrees
        h0: Reference altitude in meters

    Returns:
        Tuple of (latitude, longitude, altitude) in degrees and meters
    """
    # Convert reference lat/lon to radians
    phi0_rad = np.deg2rad(phi0_deg)

    # Calculate offsets
    delta_phi = x / get_meridional_radius(phi0_rad)
    delta_lam = y / (get_prime_vertical_radius(phi0_rad) * np.cos(phi0_rad))

    # Add to original (in degrees)
    phi = phi0_deg + np.rad2deg(delta_phi)
    lam = lam0_deg + np.rad2deg(delta_lam)

    # In NED, positive Z is down, so we subtract Z from altitude
    h = h0 - z
    return phi.item(), lam.item(), h.item()


def get_calibration_matrix(K_coefs: list, scale: tuple = (1, 1)) -> np.ndarray:
    """Construct camera intrinsic calibration matrix from coefficients.

    Args:
        K_coefs: List of [fx, fy, cx, cy] intrinsic parameters
        scale: Optional (sx, sy) scaling factors for image resizing

    Returns:
        3x3 camera calibration matrix K
    """
    sx, sy = scale

    # Construct Camera Matrix
    K = np.array([[K_coefs[0] * sx, 0, K_coefs[2] * sx],
                  [0, K_coefs[1] * sy, K_coefs[3] * sy],
                  [0, 0, 1]])
    return K


def project_fov_to_ground(lat, lon, alt, rel_alt, pitch, yaw, roll, K_coefs, pxl_vectors, scale=(1, 1), verbose=-1):
    # 1. Get Rotation Matrix (Body -> NED)
    # Note: Ensure pitch/roll/yaw match the drone's IMU frame (Nose Forward)
    R_body_to_ned = get_rotation_matrix(pitch, yaw, roll)
    K = get_calibration_matrix(K_coefs, scale)

    # 3. Define Pixel Vectors (Homogeneous)
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    pxl_vectors = pxl_vectors.T

    # 4. Convert Pixels to Normalized Camera Frame (Ray direction)
    # Current Frame: Camera Frame (X-Right, Y-Down, Z-Forward)
    dir_vectors_cam = np.linalg.inv(K) @ pxl_vectors

    # 5. Transform Camera Frame to Body Frame
    # Camera: Z is Forward, X is Right, Y is Down
    # Body (NED): X is Forward, Y is Right, Z is Down
    # Transform: X_body = Z_cam, Y_body = X_cam, Z_body = Y_cam
    R_cam_to_body = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])

    dir_vectors_body = R_cam_to_body @ dir_vectors_cam

    # 6. Transform Body Frame to NED Frame
    # Use R, not inv(R), because we are projecting OUT to the world
    dir_vectors_ned = R_body_to_ned @ dir_vectors_body

    # 7. Intersect with Ground Plane
    # We want to find a scalar 'd' such that: P_ned = d * V_ned
    # And P_ned[2] (Z component) == rel_alt (Distance down to ground)
    # So d = rel_alt / V_ned[2]

    # Transpose for easier iteration (4, 3)
    dir_vectors_ned = dir_vectors_ned.T

    # Scale vectors to hit the ground plane
    # Warning: division by zero if looking perfectly horizontal or up
    scale_factors = rel_alt / dir_vectors_ned[:, 2]

    # Broadcasting the scale factor to X, Y, Z
    points_vectors_Q_NED = dir_vectors_ned * scale_factors[:, np.newaxis]
    if verbose > 0:
        print("NED Offsets (X, Y, Z):\n", points_vectors_Q_NED)

    # 8. Convert NED offsets to Geodetic
    points_vectors_Q_coord = []
    for i in range(len(points_vectors_Q_NED)):
        phi, lam, h = get_lat_lon_alt_from_NED_dep(
            points_vectors_Q_NED[i, 0],
            points_vectors_Q_NED[i, 1],
            points_vectors_Q_NED[i, 2],
            lat, lon, alt)
        points_vectors_Q_coord.append([phi, lam, h])

    return points_vectors_Q_coord


def compute_frames_fov(metadata_df, img_width, img_height, camera_intrinsics):

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
        fov_coords = project_fov_to_ground(
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

    return fov_wgs84_list, rotation_matrix_list, [[global_min_lat, global_min_lon], [global_max_lat, global_max_lon]]


# --- usage example ---
if __name__ == "__main__":
    lat, lon, alt = 48.700130, 2.026754, 189.220
    rel_alt = 54.698
    pitch, yaw, roll = -24.3, -139.1, 0
    K_coefs = [1387.265, 1387.644, 956.787, 537.653]
    image_shape = (1920, 1080)

    coords = project_fov_to_ground(lat, lon, alt, rel_alt, pitch,
                                   yaw, roll, image_shape, K_coefs)

    print("\nCalculated Coordinates (Lat, Lon, Alt):")
    for pt in coords:
        print(pt)
