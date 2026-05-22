import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH SETUP
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "perspective_detection_experiment")

RAW_DETECTION_DIR = os.path.join(OUTPUT_DIR, "raw_detection")
PERSPECTIVE_DIR = os.path.join(OUTPUT_DIR, "perspective_best")
PERSPECTIVE_DETECTION_DIR = os.path.join(OUTPUT_DIR, "perspective_best_detection")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DETECTION_DIR, exist_ok=True)
os.makedirs(PERSPECTIVE_DIR, exist_ok=True)
os.makedirs(PERSPECTIVE_DETECTION_DIR, exist_ok=True)


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

# Best setting selected from the 5-frame debug experiment
BEST_FOCAL_SCALE = 0.25
BEST_CANVAS_SCALE = 1.5
BEST_YAW_DEG = 0
BEST_PITCH_DEG = 0

# To avoid processing too many accidental images, we only use files named frame_*.jpg/png.
FRAME_PREFIX = "frame_"
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Optional limit.
# Set to None to process all frames.
# Set to 50 or 100 first if you want a quick test.
MAX_FRAMES = None


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
# 4. FIND FRAMES
# ============================================================

def find_all_frames():
    """
    Find only original raw frame images with names like:
        frame_000000.jpg
        frame_000030.jpg
        frame_002940.jpg

    Exclude generated files like:
        frame_000000_down.jpg
        frame_000000_front.jpg
        frame_000000_perspective.jpg
        frame_000000_detected.jpg
    """

    import re

    frames = []

    raw_frame_pattern = re.compile(r"^frame_\d{6}\.(jpg|jpeg|png)$", re.IGNORECASE)

    skip_dirs = {
        ".git",
        "__pycache__",
        "results",
    }

    for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
        # Do not walk into results or cache folders
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs
        ]

        for filename in filenames:
            if raw_frame_pattern.match(filename):
                frames.append(os.path.join(dirpath, filename))

    frames = sorted(frames)

    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    return frames


# ============================================================
# 5. LOAD OMNIDIRECTIONAL CALIBRATION
# ============================================================

def load_omni_calibration():
    possible_files = [
        os.path.join(RESULTS_DIR, "omni_calibration.npz"),
        os.path.join(RESULTS_DIR, "omnidir_calibration.npz"),
        os.path.join(RESULTS_DIR, "omnidirectional_calibration.npz"),

        os.path.join(RESULTS_DIR, "calibrations", "omni_calibration.npz"),
        os.path.join(RESULTS_DIR, "calibrations", "omnidir_calibration.npz"),
        os.path.join(RESULTS_DIR, "calibrations", "omnidirectional_calibration.npz"),

        os.path.join(SRC_DIR, "omni_calibration.npz"),
        os.path.join(SRC_DIR, "omnidir_calibration.npz"),
        os.path.join(SRC_DIR, "omnidirectional_calibration.npz"),
    ]

    for path in possible_files:
        if os.path.exists(path):
            print(f"\nLoaded omnidirectional calibration from: {path}")

            data = np.load(path)
            print("Available keys in calibration file:", list(data.keys()))

            if "K" in data:
                K = data["K"]
            elif "camera_matrix" in data:
                K = data["camera_matrix"]
            elif "mtx" in data:
                K = data["mtx"]
            else:
                raise KeyError(
                    "Could not find camera matrix. Expected key 'K', 'camera_matrix', or 'mtx'."
                )

            if "D" in data:
                D = data["D"]
            elif "dist" in data:
                D = data["dist"]
            elif "dist_coeffs" in data:
                D = data["dist_coeffs"]
            else:
                raise KeyError(
                    "Could not find distortion coefficients. Expected key 'D', 'dist', or 'dist_coeffs'."
                )

            if "xi" in data:
                xi = data["xi"]
            else:
                raise KeyError("Could not find xi in calibration file.")

            K = np.array(K, dtype=np.float64)
            D = np.array(D, dtype=np.float64).reshape(-1, 1)

            xi_value = float(np.array(xi).ravel()[0])
            xi = np.array([[xi_value]], dtype=np.float64)

            print("\nCalibration loaded:")
            print("K shape:", K.shape)
            print("D shape:", D.shape)
            print("xi:", xi)
            print("xi shape:", xi.shape)

            if "rms" in data:
                print("RMS:", data["rms"])

            if "image_size" in data:
                print("Calibration image size:", data["image_size"])

            if "frame_count" in data:
                print("Calibration frame count:", data["frame_count"])

            return K, D, xi

    raise FileNotFoundError(
        "Could not find omnidirectional calibration file.\n\n"
        "I searched these paths:\n" +
        "\n".join(possible_files)
    )


# ============================================================
# 6. DETECT CHARUCO
# ============================================================

def detect_charuco(image):
    """
    Detect ArUco markers and ChArUco corners.
    Returns:
      visualization image,
      marker count,
      ChArUco corner count
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    marker_count = 0
    charuco_count = 0

    vis = image.copy()

    if ids is not None and len(ids) > 0:
        marker_count = len(ids)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_ids is not None:
            charuco_count = len(charuco_ids)
            cv2.aruco.drawDetectedCornersCharuco(
                vis,
                charuco_corners,
                charuco_ids,
                cornerColor=(255, 0, 0)
            )

    return vis, marker_count, charuco_count


# ============================================================
# 7. ROTATION AND PERSPECTIVE RECTIFICATION
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


def make_perspective(
    image,
    K,
    D,
    xi,
    yaw_deg=0,
    pitch_deg=0,
    focal_scale=0.25,
    canvas_scale=1.5
):
    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64).reshape(-1, 1)
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
        D,
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

    return perspective


# ============================================================
# 8. STATUS HELPER
# ============================================================

def get_status(raw_charuco, perspective_charuco):
    gain = perspective_charuco - raw_charuco

    if gain > 0:
        return "improved"
    elif gain == 0:
        return "same"
    else:
        return "worse"


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():
    print("Starting perspective detection experiment...")
    print("Project directory:", PROJECT_DIR)
    print("Output directory:", OUTPUT_DIR)

    frame_paths = find_all_frames()

    print(f"\nFound {len(frame_paths)} frame images.")

    if len(frame_paths) == 0:
        raise FileNotFoundError(
            "No frame_*.jpg/png images found in the project. "
            "Please check where your extracted frames are saved."
        )

    K, D, xi = load_omni_calibration()

    rows = []

    for i, frame_path in enumerate(frame_paths, start=1):
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]

        print(f"\n[{i}/{len(frame_paths)}] Processing {frame_name}")
        print("Path:", frame_path)

        image = cv2.imread(frame_path)

        if image is None:
            print("Could not read image, skipping.")
            continue

        # ----------------------------------------------------
        # Raw detection
        # ----------------------------------------------------
        raw_vis, raw_marker_count, raw_charuco_count = detect_charuco(image)

        raw_det_out = os.path.join(
            RAW_DETECTION_DIR,
            f"{frame_name}_raw_detection.jpg"
        )
        cv2.imwrite(raw_det_out, raw_vis)

        # ----------------------------------------------------
        # Best perspective rectification
        # ----------------------------------------------------
        perspective = make_perspective(
            image,
            K,
            D,
            xi,
            yaw_deg=BEST_YAW_DEG,
            pitch_deg=BEST_PITCH_DEG,
            focal_scale=BEST_FOCAL_SCALE,
            canvas_scale=BEST_CANVAS_SCALE
        )

        perspective_out = os.path.join(
            PERSPECTIVE_DIR,
            f"{frame_name}_perspective_best_focal_025.jpg"
        )
        cv2.imwrite(perspective_out, perspective)

        # ----------------------------------------------------
        # Perspective detection
        # ----------------------------------------------------
        perspective_vis, perspective_marker_count, perspective_charuco_count = detect_charuco(
            perspective
        )

        perspective_det_out = os.path.join(
            PERSPECTIVE_DETECTION_DIR,
            f"{frame_name}_perspective_best_focal_025_detection.jpg"
        )
        cv2.imwrite(perspective_det_out, perspective_vis)

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------
        marker_gain = perspective_marker_count - raw_marker_count
        charuco_gain = perspective_charuco_count - raw_charuco_count
        status = get_status(raw_charuco_count, perspective_charuco_count)

        print(
            f"Raw: markers={raw_marker_count}, charuco={raw_charuco_count} | "
            f"Perspective: markers={perspective_marker_count}, charuco={perspective_charuco_count} | "
            f"Gain={charuco_gain} | {status}"
        )

        rows.append({
            "frame": frame_name,
            "source_path": frame_path,

            "raw_markers": raw_marker_count,
            "raw_charuco_corners": raw_charuco_count,

            "perspective_markers": perspective_marker_count,
            "perspective_charuco_corners": perspective_charuco_count,

            "marker_gain": marker_gain,
            "charuco_gain": charuco_gain,
            "status": status,

            "raw_detection_image": raw_det_out,
            "perspective_image": perspective_out,
            "perspective_detection_image": perspective_det_out,
        })

    summary = pd.DataFrame(rows)

    summary_path = os.path.join(OUTPUT_DIR, "perspective_detection_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n======================================")
    print("DONE")
    print("======================================")
    print("Summary saved to:", summary_path)

    if len(summary) > 0:
        print("\nOverall totals:")
        print("Raw markers:", summary["raw_markers"].sum())
        print("Raw ChArUco corners:", summary["raw_charuco_corners"].sum())
        print("Perspective markers:", summary["perspective_markers"].sum())
        print("Perspective ChArUco corners:", summary["perspective_charuco_corners"].sum())

        print("\nFrame status counts:")
        print(summary["status"].value_counts())

        print("\nTop improved frames:")
        print(
            summary.sort_values("charuco_gain", ascending=False)
            [["frame", "raw_charuco_corners", "perspective_charuco_corners", "charuco_gain", "status"]]
            .head(20)
        )


if __name__ == "__main__":
    main()
