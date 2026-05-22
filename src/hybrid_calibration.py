import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH SETUP
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

HYBRID_DIR = os.path.join(RESULTS_DIR, "hybrid_detection_selection")
HYBRID_USEFUL_CSV = os.path.join(HYBRID_DIR, "hybrid_useful_frames.csv")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "hybrid_omnidir_calibration")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_mapped_points")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


# ============================================================
# 2. BEST PERSPECTIVE SETTINGS
# ============================================================

BEST_FOCAL_SCALE = 0.25
BEST_CANVAS_SCALE = 1.5
BEST_YAW_DEG = 0
BEST_PITCH_DEG = 0


# ============================================================
# 3. CHARUCO BOARD CONFIGURATION
# ============================================================
# Your custom board:
# - ArUco dictionary: DICT_4X4_1000
# - Board squares: 7 x 7
# - Inner ChArUco corners: 6 x 6 = 36
# - square_len = 0.14285714285714285
# - marker_len = 0.10

ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_1000

SQUARES_X = 7
SQUARES_Y = 7

SQUARE_LENGTH = 0.14285714285714285
MARKER_LENGTH = 0.10

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)


# ============================================================
# 4. LOAD INITIAL OMNIDIR CALIBRATION
# ============================================================

def load_initial_omni_calibration():
    possible_files = [
        os.path.join(RESULTS_DIR, "omni_calibration.npz"),
        os.path.join(RESULTS_DIR, "omnidir_calibration.npz"),
        os.path.join(RESULTS_DIR, "omnidirectional_calibration.npz"),
        os.path.join(RESULTS_DIR, "calibrations", "omni_calibration.npz"),
        os.path.join(RESULTS_DIR, "calibrations", "omnidir_calibration.npz"),
        os.path.join(SRC_DIR, "omni_calibration.npz"),
        os.path.join(SRC_DIR, "omnidir_calibration.npz"),
    ]

    calib_path = None

    for path in possible_files:
        if os.path.exists(path):
            calib_path = path
            break

    if calib_path is None:
        raise FileNotFoundError(
            "Could not find initial omnidirectional calibration file.\n"
            "Expected one of:\n" + "\n".join(possible_files)
        )

    data = np.load(calib_path)

    print("\nLoaded initial omnidir calibration from:")
    print(calib_path)
    print("Available keys:", list(data.keys()))

    if "camera_matrix" in data:
        K = data["camera_matrix"]
    elif "K" in data:
        K = data["K"]
    elif "mtx" in data:
        K = data["mtx"]
    else:
        raise KeyError("Could not find camera matrix. Expected camera_matrix, K, or mtx.")

    if "dist_coeffs" in data:
        D = data["dist_coeffs"]
    elif "D" in data:
        D = data["D"]
    elif "dist" in data:
        D = data["dist"]
    else:
        raise KeyError("Could not find distortion coefficients. Expected dist_coeffs, D, or dist.")

    if "xi" not in data:
        raise KeyError("Could not find xi in calibration file.")

    xi = data["xi"]

    K = np.array(K, dtype=np.float64).reshape(3, 3)

    # For omnidir calibration OpenCV is sometimes strict.
    # For initUndistortRectifyMap: D works as (4, 1)
    # For calibrate with USE_GUESS: D is safer as (1, 4)
    D_map = np.array(D, dtype=np.float64).reshape(-1, 1)
    D_calib = np.array(D, dtype=np.float64).reshape(1, 4)

    xi_value = float(np.array(xi).ravel()[0])
    xi = np.array([[xi_value]], dtype=np.float64)

    print("K shape:", K.shape)
    print("D_map shape:", D_map.shape)
    print("D_calib shape:", D_calib.shape)
    print("xi:", xi)
    print("xi shape:", xi.shape)

    if "rms" in data:
        print("Initial RMS:", data["rms"])

    return K, D_map, D_calib, xi


# ============================================================
# 5. DETECT CHARUCO
# ============================================================

def detect_charuco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    vis = image.copy()

    if ids is None or len(ids) == 0:
        return vis, None, None, 0, 0

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    marker_count = len(ids)

    if charuco_ids is None or charuco_corners is None:
        return vis, None, None, marker_count, 0

    charuco_count = len(charuco_ids)

    cv2.aruco.drawDetectedCornersCharuco(
        vis,
        charuco_corners,
        charuco_ids,
        cornerColor=(255, 0, 0)
    )

    return vis, charuco_corners, charuco_ids, marker_count, charuco_count


# ============================================================
# 6. BOARD OBJECT POINTS FROM CHARUCO IDS
# ============================================================

def get_board_object_points(charuco_ids):
    """
    Convert detected ChArUco IDs to 3D board coordinates.

    Required output shape for cv2.omnidir.calibrate:
        (1, N, 3)
    """

    all_corners = board.getChessboardCorners()
    all_corners = np.array(all_corners, dtype=np.float64)

    object_points = []

    for corner_id in charuco_ids.flatten():
        object_points.append(all_corners[int(corner_id)])

    object_points = np.array(object_points, dtype=np.float64).reshape(1, -1, 3)

    return object_points


# ============================================================
# 7. PERSPECTIVE RECTIFICATION MAP
# ============================================================

def rotation_from_yaw_pitch(yaw_deg=0, pitch_deg=0):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float64)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float64)

    return Rx @ Ry


def make_perspective_and_maps(
    image,
    K,
    D_map,
    xi,
    yaw_deg=0,
    pitch_deg=0,
    focal_scale=0.25,
    canvas_scale=1.5
):
    K = np.array(K, dtype=np.float64).reshape(3, 3)
    D_map = np.array(D_map, dtype=np.float64).reshape(-1, 1)
    xi = np.array(xi, dtype=np.float64).reshape(1, 1)

    h, w = image.shape[:2]

    out_w = int(canvas_scale * w)
    out_h = int(canvas_scale * h)

    R = rotation_from_yaw_pitch(yaw_deg, pitch_deg)

    fx = focal_scale * w
    fy = focal_scale * w
    cx = out_w / 2
    cy = out_h / 2

    P = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K,
        D_map,
        xi,
        R,
        P,
        (out_w, out_h),
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    perspective = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return perspective, map1, map2


# ============================================================
# 8. MAP PERSPECTIVE POINTS BACK TO RAW IMAGE
# ============================================================

def bilinear_sample_map(map_img, x, y):
    h, w = map_img.shape[:2]

    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return np.nan

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    value = (
        (1 - dx) * (1 - dy) * map_img[y0, x0]
        + dx * (1 - dy) * map_img[y0, x1]
        + (1 - dx) * dy * map_img[y1, x0]
        + dx * dy * map_img[y1, x1]
    )

    return value


def map_perspective_corners_to_raw(charuco_corners, map1, map2):
    """
    charuco_corners are detected in the perspective image.

    map1/map2 tell for each perspective pixel where it came from
    in the raw image.

    Returns:
        mapped_points: shape (N, 2)
        valid_mask: shape (N,)
    """

    mapped_points = []

    for pt in charuco_corners.reshape(-1, 2):
        x_p, y_p = float(pt[0]), float(pt[1])

        x_raw = bilinear_sample_map(map1, x_p, y_p)
        y_raw = bilinear_sample_map(map2, x_p, y_p)

        mapped_points.append([x_raw, y_raw])

    mapped_points = np.array(mapped_points, dtype=np.float64)

    valid_mask = ~np.isnan(mapped_points).any(axis=1)

    mapped_points = mapped_points[valid_mask]

    return mapped_points, valid_mask


# ============================================================
# 9. DRAW MAPPED POINTS ON RAW IMAGE
# ============================================================

def draw_mapped_points(raw_image, mapped_points, frame_name, selected_source):
    vis = raw_image.copy()

    points = np.array(mapped_points, dtype=np.float64).reshape(-1, 2)

    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))

        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    out_path = os.path.join(
        DEBUG_DIR,
        f"{frame_name}_{selected_source}_mapped_points_on_raw.jpg"
    )

    cv2.imwrite(out_path, vis)

    return out_path


# ============================================================
# 10. COLLECT HYBRID CORRESPONDENCES
# ============================================================

def collect_hybrid_correspondences():
    if not os.path.exists(HYBRID_USEFUL_CSV):
        raise FileNotFoundError(
            f"Could not find hybrid useful frames CSV:\n{HYBRID_USEFUL_CSV}\n\n"
            "Run src/hybrid_detection_selection.py first."
        )

    df = pd.read_csv(HYBRID_USEFUL_CSV)

    K_init, D_map_init, D_calib_init, xi_init = load_initial_omni_calibration()

    object_points_all = []
    image_points_all = []

    frame_rows = []

    image_size = None

    for idx, row in df.iterrows():
        frame_name = row["frame"]
        source_path = row["source_path"]
        selected_source = row["selected_source"]

        print(f"\n[{idx + 1}/{len(df)}] Processing {frame_name} ({selected_source})")

        image = cv2.imread(source_path)

        if image is None:
            print("Could not read image, skipping.")
            continue

        h, w = image.shape[:2]
        image_size = (w, h)

        # ----------------------------------------------------
        # Case A: selected raw detection
        # ----------------------------------------------------
        if selected_source == "raw":
            vis, charuco_corners, charuco_ids, marker_count, charuco_count = detect_charuco(image)

            if charuco_corners is None or charuco_ids is None:
                print("No raw ChArUco corners found, skipping.")
                continue

            image_points = np.array(charuco_corners, dtype=np.float64).reshape(1, -1, 2)
            object_points = get_board_object_points(charuco_ids)

            debug_path = draw_mapped_points(
                image,
                image_points,
                frame_name,
                selected_source="raw"
            )

        # ----------------------------------------------------
        # Case B: selected perspective detection
        # ----------------------------------------------------
        elif selected_source == "perspective":
            perspective, map1, map2 = make_perspective_and_maps(
                image,
                K_init,
                D_map_init,
                xi_init,
                yaw_deg=BEST_YAW_DEG,
                pitch_deg=BEST_PITCH_DEG,
                focal_scale=BEST_FOCAL_SCALE,
                canvas_scale=BEST_CANVAS_SCALE
            )

            vis, charuco_corners_p, charuco_ids_p, marker_count, charuco_count = detect_charuco(
                perspective
            )

            if charuco_corners_p is None or charuco_ids_p is None:
                print("No perspective ChArUco corners found, skipping.")
                continue

            mapped_points, valid_mask = map_perspective_corners_to_raw(
                charuco_corners_p,
                map1,
                map2
            )

            charuco_ids_valid = charuco_ids_p.reshape(-1)[valid_mask].reshape(-1, 1)

            if len(mapped_points) < 4:
                print("Too few valid mapped points after back-mapping, skipping.")
                continue

            image_points = np.array(mapped_points, dtype=np.float64).reshape(1, -1, 2)
            object_points = get_board_object_points(charuco_ids_valid)

            debug_path = draw_mapped_points(
                image,
                image_points,
                frame_name,
                selected_source="perspective"
            )

        else:
            print("Unknown selected source, skipping:", selected_source)
            continue

        # ----------------------------------------------------
        # Shape safety checks
        # ----------------------------------------------------
        if image_points.ndim != 3 or object_points.ndim != 3:
            print("Invalid point dimensions, skipping.")
            print("image_points shape:", image_points.shape)
            print("object_points shape:", object_points.shape)
            continue

        if image_points.shape[1] != object_points.shape[1]:
            print("Point count mismatch, skipping.")
            print("image_points shape:", image_points.shape)
            print("object_points shape:", object_points.shape)
            continue

        if image_points.shape[1] < 4:
            print("Too few points for calibration, skipping.")
            continue

        object_points_all.append(object_points.astype(np.float64))
        image_points_all.append(image_points.astype(np.float64))

        frame_rows.append({
            "frame": frame_name,
            "source_path": source_path,
            "selected_source": selected_source,
            "num_points": int(image_points.shape[1]),
            "debug_mapped_points_image": debug_path,
        })

        print("Accepted points:", image_points.shape[1])

    frame_summary = pd.DataFrame(frame_rows)

    return (
        object_points_all,
        image_points_all,
        image_size,
        frame_summary,
        K_init,
        D_calib_init,
        xi_init
    )


# ============================================================
# 11. OMNIDIR CALIBRATION
# ============================================================

def run_omnidir_calibration(
    object_points_all,
    image_points_all,
    image_size,
    K_init,
    D_calib_init,
    xi_init
):
    if len(object_points_all) < 3:
        raise ValueError(
            "Not enough frames for calibration. Need at least 3 valid frames."
        )

    print("\nRunning hybrid omnidirectional calibration...")
    print("Number of frames:", len(object_points_all))
    print("Image size:", image_size)

    print("\nChecking point array shapes:")
    print("First object points shape:", object_points_all[0].shape)
    print("First image points shape:", image_points_all[0].shape)

    K = np.array(K_init, dtype=np.float64).reshape(3, 3).copy()
    D = np.array(D_calib_init, dtype=np.float64).reshape(1, 4).copy()
    xi = np.array(xi_init, dtype=np.float64).reshape(1, 1).copy()

    flags = cv2.omnidir.CALIB_USE_GUESS

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    rms, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(
        object_points_all,
        image_points_all,
        image_size,
        K,
        xi,
        D,
        flags,
        criteria
    )

    return rms, K, xi, D, rvecs, tvecs, idx


# ============================================================
# 12. MAIN
# ============================================================

def main():
    print("Starting hybrid omnidirectional calibration...")
    print("Using:", HYBRID_USEFUL_CSV)

    (
        object_points_all,
        image_points_all,
        image_size,
        frame_summary,
        K_init,
        D_calib_init,
        xi_init
    ) = collect_hybrid_correspondences()

    frame_summary_path = os.path.join(OUTPUT_DIR, "hybrid_calibration_frames.csv")
    frame_summary.to_csv(frame_summary_path, index=False)

    print("\nCollected correspondence frames:", len(object_points_all))
    print("Frame summary saved to:", frame_summary_path)

    if len(object_points_all) == 0:
        raise RuntimeError("No valid correspondences collected.")

    rms, K, xi, D, rvecs, tvecs, idx = run_omnidir_calibration(
        object_points_all,
        image_points_all,
        image_size,
        K_init,
        D_calib_init,
        xi_init
    )

    output_npz = os.path.join(OUTPUT_DIR, "hybrid_omni_calibration.npz")

    np.savez(
        output_npz,
        rms=rms,
        camera_matrix=K,
        xi=xi,
        dist_coeffs=D,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        idx=idx,
        image_size=np.array(image_size),
        frame_count=len(object_points_all),
    )

    print("\n======================================")
    print("DONE")
    print("======================================")
    print("Hybrid omnidir calibration saved to:", output_npz)
    print("RMS:", rms)
    print("K:\n", K)
    print("xi:", xi)
    print("D:", D.ravel())
    print("idx:", idx.ravel() if idx is not None else idx)


if __name__ == "__main__":
    main()