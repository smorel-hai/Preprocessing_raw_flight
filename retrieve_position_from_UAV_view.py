import numpy as np


def get_R(pitch, yaw, roll):
    """
    Returns the Rotation Matrix from Body Frame to NED Frame.
    Order: Yaw -> Pitch -> Roll (Standard Aerospace)
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


def Rn(phi, a=6378137.0, e2=0.0066943799901413165):
    """Meridional Radius of Curvature"""
    return (a*(1 - e2)/np.power((1 - e2*np.sin(phi)**2), 3/2))


def Re(phi, a=6378137.0, e2=0.0066943799901413165):
    """Prime Vertical Radius of Curvature"""
    return a / np.sqrt(1 - e2*np.sin(phi)**2)


def get_lat_lon_alt_from_NED_dep(x, y, z, phi0_deg, lam0_deg, h0):
    # Convert reference lat/lon to radians
    phi0_rad = np.deg2rad(phi0_deg)

    # Calculate offsets
    delta_phi = x / Rn(phi0_rad)
    delta_lam = y / (Re(phi0_rad) * np.cos(phi0_rad))

    # Add to original (in degrees)
    phi = phi0_deg + np.rad2deg(delta_phi)
    lam = lam0_deg + np.rad2deg(delta_lam)

    # In NED, positive Z is down, so we subtract Z from altitude
    h = h0 - z
    return phi.item(), lam.item(), h.item()


def process(lat, lon, alt, rel_alt, pitch, yaw, roll, K_coefs, pxl_vectors, scale=(1, 1), verbose=-1):
    # 1. Get Rotation Matrix (Body -> NED)
    # Note: Ensure pitch/roll/yaw match the drone's IMU frame (Nose Forward)
    R_body_to_ned = get_R(pitch, yaw, roll)
    sx, sy = scale

    # 2. Construct Camera Matrix
    K = np.array([[K_coefs[0] * sx, 0, K_coefs[2] * sx],
                  [0, K_coefs[1] * sy, K_coefs[3] * sy],
                  [0, 0, 1]])

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
    T_cam_to_body = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])

    dir_vectors_body = T_cam_to_body @ dir_vectors_cam

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


# --- usage example ---
if __name__ == "__main__":
    lat, lon, alt = 48.700130, 2.026754, 189.220
    rel_alt = 54.698
    pitch, yaw, roll = -24.3, -139.1, 0
    K_coefs = [1387.265, 1387.644, 956.787, 537.653]
    image_shape = (1920, 1080)

    coords = process(lat, lon, alt, rel_alt, pitch,
                     yaw, roll, image_shape, K_coefs)

    print("\nCalculated Coordinates (Lat, Lon, Alt):")
    for pt in coords:
        print(pt)
