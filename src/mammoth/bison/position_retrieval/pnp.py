import cv2
import numpy as np


PNP_SOLVERS = {
    # --- The Modern Standard ---
    "SQPNP": cv2.SOLVEPNP_SQPNP,          # Best general purpose (global optimum, robust)

    # --- The Classic / Default ---
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,  # Default (Levenberg-Marquardt). Needs good init, slow but precise.

    # --- For Small Sets (N=3 or N=4) ---
    "P3P": cv2.SOLVEPNP_P3P,              # Classic algebraic solver for exactly 4 points. Fast, but jittery noise.
    "AP3P": cv2.SOLVEPNP_AP3P,            # "Advanced" P3P (Available in newer OpenCV). Generally more stable than P3P.

    # --- For Coplanar Points (Flat Objects) ---
    "IPPE": cv2.SOLVEPNP_IPPE,            # Fastest/Most robust for flat objects (Z=0). Fails if non-planar.
    # "IPPE_SQUARE": cv2.SOLVEPNP_IPPE_SQUARE,  # Special case for marker tracking (ArUco markers) ### Comment because too specific

    # --- For Large Sets (N > 4) ---
    # "Efficient PnP". Fast non-iterative O(n) solution. Good initial guess for Iterative.
    "EPNP": cv2.SOLVEPNP_EPNP,

    # --- Older / Less Common --- #### Comment because replaced by EPNP
    # "Universal PnP". Attempts to solve camera focal length too (unstable usually).
    # "UPNP": cv2.SOLVEPNP_UPNP,
    # "DLS": cv2.SOLVEPNP_DLS               # "Direct Least Squares". Often unstable, rarely used now vs SQPNP.
}


def get_camera_position_robust(image_points: np.ndarray, object_points: np.ndarray,
                               camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None,
                               solver_type: str = 'SQPNP', verbose: int = 1) -> tuple:
    """Robustly estimate camera position using Perspective-n-Point algorithm.

    This function solves for the camera pose (position and orientation) given:
    - Known 3D points in world coordinates (object_points)
    - Their corresponding 2D projections in the image (image_points)
    - Camera calibration parameters

    The function uses a centering trick to improve numerical stability with
    large coordinate values (e.g., Web Mercator coordinates).

    Args:
        image_points: Nx2 array of pixel coordinates (x, y)
        object_points: Nx3 array of 3D world coordinates (x, y, z)
        camera_matrix: 3x3 camera intrinsic matrix K
        dist_coeffs: Lens distortion coefficients (default: None/zero)
        solver_type: PnP algorithm to use (default: 'SQPNP')
            Options: 'SQPNP', 'ITERATIVE', 'IPPE', 'EPNP', 'P3P', 'AP3P'
        verbose: Verbosity level (0=silent, 1=normal)

    Returns:
        Tuple of (camera_position_global, rotation_vector)
            - camera_position_global: 3D camera position in world coordinates
            - rotation_vector: Rodrigues rotation vector

    Raises:
        ValueError: If PnP solution fails to converge
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    if image_points.dtype == np.integer:
        image_points = image_points.astype(np.floating)

    # --- SELECT THE CORRECT FLAG ---
    # Default to SQPNP (Best for small sets of points in OpenCV > 4.5.3)
    pnp_flag = PNP_SOLVERS[solver_type]

    # --- CENTERING TRICK (Crucial for Web Mercator/GPS) ---
    centroid = np.mean(object_points, axis=0)
    local_object_points = object_points - centroid

    # Check if points are Coplanar (e.g., all on a map with Z=0)
    # If Z variance is near zero, use IPPE (Iterative for Planar Pose Estimation)
    # if np.std(local_object_points[:, 2]) < 1e-5:
    #     # print("Points appear coplanar (flat). Using IPPE solver.")
    #     pnp_flag = cv2.SOLVEPNP_IPPE

    try:

        success, rvec, tvec = cv2.solvePnP(
            local_object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=pnp_flag
        )

    except cv2.error:
        # Fallback for older OpenCV versions or if SQPNP fails
        if verbose >= 1:
            print("SQPNP/IPPE failed. Retrying with EPNP...")
        pnp_flag = cv2.SOLVEPNP_EPNP

        success, rvec, tvec = cv2.solvePnP(
            local_object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=pnp_flag
        )

    if not success:
        raise ValueError("PnP solution failed to converge.")

    # --- Calculate Camera Position ---
    # R, _ = cv2.Rodrigues(rvec)
    # camera_pos_local = -np.matrix(R).T @ np.matrix(tvec)
    np_rodrigues = np.asarray(rvec[:, :], np.float64)
    rmat = cv2.Rodrigues(np_rodrigues)[0]
    camera_position = -np.matrix(rmat).T @ np.matrix(tvec)
    camera_position_global = camera_position.A1 + centroid

    return camera_position_global, rvec


# --- TEST ---
if __name__ == '__main__':
    # Standard K
    K = np.array([[1.38726502e+03, 0.00000000e+00, 9.56787476e+02],
                  [0.00000000e+00, 1.38764439e+03, 5.37653682e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    image_shape = (1920, 1080)
    W, H = image_shape
    # 4 Points in Web Mercator (e.g., 3 on ground, 1 with height, or all on ground)
    mercator_pts = np.array([[287831.95187619, 6307449.73545761, 1.07418000e+02],
                            [287616.13656306, 6306851.84042464, 1.07418000e+02],
                            [287496.69049708, 6307571.45204713, 1.07418000e+02],
                             [287280.87518394, 6306973.54837845, 1.07418000e+02]
                             ], dtype=np.float64)

    image_pts = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32)

    pos, rot = get_camera_position_robust(image_pts, mercator_pts, K)
    print(f"Camera Position: {pos}")
