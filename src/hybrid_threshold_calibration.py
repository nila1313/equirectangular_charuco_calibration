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
HYBRID_SUMMARY_CSV = os.path.join(HYBRID_DIR, "hybrid_detection_summary.csv")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "hybrid_threshold_calibration")
DEBUG_ROOT_DIR = os.path.join(OUTPUT_DIR, "debug_mapped_points")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_ROOT_DIR, exist_ok=True)


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

BEST_FOCAL_SCALE = 0.25
BEST_CANVAS_SCALE = 1.5
BEST_YAW_DEG = 0
BEST_PITCH_DEG = 0

# Try stricter and stricter calibration frame filters
THRESHOLDS = [4, 6, 8, 10, 12]

# Optional: limit number of frames per threshold for debugging.
# Keep None for final experiment.
MAX_FRAMES_PER_THRESHOLD = None


# ============================================================
# 3. CHARUCO BOARD CONFIGURATION
# ============================================================

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
        raise KeyError("Could not find camera matrix.")

    if "dist_coeffs" in data:
        D = data["dist_coeffs"]
    elif "D" in data:
        D = data["D"]
    elif "dist" in data:
        D = data["dist"]
    else:
        raise KeyError("Could not find distortion coefficients.")

    if "xi" not in data:
        raise KeyError("Could not find xi.")

    xi = data["xi"]

    K = np.array(K, dtype=np.float64).reshape(3, 3)

    D_map = np.array(D, dtype=np.float64).reshape(-1, 1)
    D_calib = np.array(D, dtype=np.float64).reshape(1, 4)

    xi_value = float(np.array(xi).ravel()[0])
    xi = np.array([[xi_value]], dtype=np.float64)

    print("Initial K shape:", K.shape)
    print("Initial D_map shape:", D_map.shape)
    print("Initial D_calib shape:", D_calib.shape)
    print("Initial xi:", xi)

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
# 6. BOARD OBJECT POINTS
# ============================================================

def get_board_object_points(charuco_ids):
    """
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
# 7. PERSPECTIVE RECTIFICATION
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
# 8. BACK-MAP PERSPECTIVE POINTS
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
# 9. DRAW DEBUG POINTS
# ============================================================

def draw_mapped_points(raw_image, mapped_points, frame_name, selected_source, threshold):
    debug_dir = os.path.join(DEBUG_ROOT_DIR, f"threshold_{threshold}")
    os.makedirs(debug_dir, exist_ok=True)

    vis = raw_image.copy()

    points = np.array(mapped_points, dtype=np.float64).reshape(-1, 2)

    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))

        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    out_path = os.path.join(
        debug_dir,
        f"{frame_name}_{selected_source}_mapped_points_on_raw.jpg"
    )

    cv2.imwrite(out_path, vis)

    return out_path


# ============================================================
# 10. COLLECT CORRESPONDENCES FOR ONE THRESHOLD
# ============================================================

def collect_correspondences_for_threshold(
    threshold,
    K_init,
    D_map_init,
    xi_init
):
    if not os.path.exists(HYBRID_SUMMARY_CSV):
        raise FileNotFoundError(
            f"Could not find hybrid summary CSV:\n{HYBRID_SUMMARY_CSV}\n\n"
            "Run src/hybrid_detection_selection.py first."
        )

    df = pd.read_csv(HYBRID_SUMMARY_CSV)

    # Select only frames that satisfy this threshold
    df = df[df["selected_charuco_corners"] >= threshold].copy()

    if MAX_FRAMES_PER_THRESHOLD is not None:
        df = df.head(MAX_FRAMES_PER_THRESHOLD)

    print(f"\nThreshold >= {threshold}: candidate frames = {len(df)}")

    object_points_all = []
    image_points_all = []
    frame_rows = []

    image_size = None

    for idx, row in df.iterrows():
        frame_name = row["frame"]
        source_path = row["source_path"]
        selected_source = row["selected_source"]

        image = cv2.imread(source_path)

        if image is None:
            print(f"Could not read {frame_name}, skipping.")
            continue

        h, w = image.shape[:2]
        image_size = (w, h)

        if selected_source == "raw":
            vis, charuco_corners, charuco_ids, marker_count, charuco_count = detect_charuco(image)

            if charuco_corners is None or charuco_ids is None:
                continue

            image_points = np.array(charuco_corners, dtype=np.float64).reshape(1, -1, 2)
            object_points = get_board_object_points(charuco_ids)

            debug_path = draw_mapped_points(
                image,
                image_points,
                frame_name,
                selected_source="raw",
                threshold=threshold
            )

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
                continue

            mapped_points, valid_mask = map_perspective_corners_to_raw(
                charuco_corners_p,
                map1,
                map2
            )

            charuco_ids_valid = charuco_ids_p.reshape(-1)[valid_mask].reshape(-1, 1)

            if len(mapped_points) < threshold:
                continue

            image_points = np.array(mapped_points, dtype=np.float64).reshape(1, -1, 2)
            object_points = get_board_object_points(charuco_ids_valid)

            debug_path = draw_mapped_points(
                image,
                image_points,
                frame_name,
                selected_source="perspective",
                threshold=threshold
            )

        else:
            continue

        if image_points.ndim != 3 or object_points.ndim != 3:
            continue

        if image_points.shape[1] != object_points.shape[1]:
            continue

        if image_points.shape[1] < threshold:
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

    frame_summary = pd.DataFrame(frame_rows)

    return object_points_all, image_points_all, image_size, frame_summary


# ============================================================
# 11. RUN OMNIDIR CALIBRATION
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
        raise ValueError("Not enough frames for calibration.")

    K = np.array(K_init, dtype=np.float64).reshape(3, 3).copy()
    D = np.array(D_calib_init, dtype=np.float64).reshape(1, 4).copy()
    xi = np.array(xi_init, dtype=np.float64).reshape(1, 1).copy()

    flags = cv2.omnidir.CALIB_USE_GUESS

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    print("First object points shape:", object_points_all[0].shape)
    print("First image points shape:", image_points_all[0].shape)

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
# 12. MAIN EXPERIMENT
# ============================================================

def main():
    print("Starting hybrid threshold calibration experiment...")
    print("Using hybrid summary:", HYBRID_SUMMARY_CSV)

    K_init, D_map_init, D_calib_init, xi_init = load_initial_omni_calibration()

    result_rows = []

    for threshold in THRESHOLDS:
        print("\n" + "=" * 60)
        print(f"Running threshold experiment: selected_charuco_corners >= {threshold}")
        print("=" * 60)

        threshold_dir = os.path.join(OUTPUT_DIR, f"threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)

        try:
            object_points_all, image_points_all, image_size, frame_summary = (
                collect_correspondences_for_threshold(
                    threshold,
                    K_init,
                    D_map_init,
                    xi_init
                )
            )

            frame_summary_path = os.path.join(
                threshold_dir,
                f"hybrid_calibration_frames_threshold_{threshold}.csv"
            )
            frame_summary.to_csv(frame_summary_path, index=False)

            frame_count = len(object_points_all)

            print(f"Collected frames for threshold {threshold}: {frame_count}")

            if frame_count < 3:
                print("Skipping calibration: not enough frames.")
                result_rows.append({
                    "threshold": threshold,
                    "candidate_frames": len(frame_summary),
                    "calibration_frames": frame_count,
                    "idx_count": 0,
                    "rms": np.nan,
                    "xi": np.nan,
                    "status": "not_enough_frames",
                    "output_npz": "",
                    "frame_summary_csv": frame_summary_path,
                })
                continue

            rms, K, xi, D, rvecs, tvecs, idx = run_omnidir_calibration(
                object_points_all,
                image_points_all,
                image_size,
                K_init,
                D_calib_init,
                xi_init
            )

            output_npz = os.path.join(
                threshold_dir,
                f"hybrid_omni_calibration_threshold_{threshold}.npz"
            )

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
                frame_count=frame_count,
                threshold=threshold,
            )

            idx_count = 0 if idx is None else len(np.array(idx).ravel())

            print(f"Threshold {threshold} DONE")
            print("RMS:", rms)
            print("xi:", xi.ravel())
            print("D:", D.ravel())
            print("idx_count:", idx_count)
            print("Saved:", output_npz)

            result_rows.append({
                "threshold": threshold,
                "candidate_frames": len(frame_summary),
                "calibration_frames": frame_count,
                "idx_count": idx_count,
                "rms": float(rms),
                "xi": float(np.array(xi).ravel()[0]),
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "d0": float(D.ravel()[0]),
                "d1": float(D.ravel()[1]),
                "d2": float(D.ravel()[2]),
                "d3": float(D.ravel()[3]),
                "status": "success",
                "output_npz": output_npz,
                "frame_summary_csv": frame_summary_path,
            })

        except Exception as e:
            print(f"Threshold {threshold} FAILED:")
            print(str(e))

            result_rows.append({
                "threshold": threshold,
                "candidate_frames": np.nan,
                "calibration_frames": np.nan,
                "idx_count": np.nan,
                "rms": np.nan,
                "xi": np.nan,
                "status": f"failed: {str(e)}",
                "output_npz": "",
                "frame_summary_csv": "",
            })

    results_df = pd.DataFrame(result_rows)

    results_csv = os.path.join(OUTPUT_DIR, "threshold_calibration_results.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n" + "=" * 60)
    print("ALL THRESHOLD EXPERIMENTS DONE")
    print("=" * 60)
    print("Results saved to:", results_csv)

    print("\nSummary:")
    print(
        results_df[
            [
                "threshold",
                "calibration_frames",
                "idx_count",
                "rms",
                "xi",
                "status",
            ]
        ]
    )


if __name__ == "__main__":
    main()
    