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

OUTPUT_DIR = os.path.join(RESULTS_DIR, "equirectangular_5frame_debug")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_detection"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "equirectangular"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "equirectangular_detection"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "perspective"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "perspective_best"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "perspective_best_detection"), exist_ok=True)


BEST_PERSPECTIVE_FOCAL_SCALE = 0.25
BEST_PERSPECTIVE_CANVAS_SCALE = 1.5
BEST_PERSPECTIVE_YAW_DEG = 0
BEST_PERSPECTIVE_PITCH_DEG = 0


# ============================================================
# 2. CHOOSE ONLY 5 FRAMES
# ============================================================
# Add the exact filenames you want here.
# The script will search for them inside the whole project folder.

FRAME_NAMES = [
    "frame_002940.jpg",
    "frame_003210.jpg",
    "frame_003540.jpg",
    "frame_003900.jpg",
    "frame_006840.jpg",

    # You can also test with the one visible in your src folder:
    # "frame_000690.jpg",
]


def find_frame_file(frame_name):
    """
    Search for a frame inside the whole project folder.
    This fixes the problem where frames may be inside:
      data/frames/
      data/raw_frames/
      results/something/
      src/
      or another subfolder.
    """

    for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
        if frame_name in filenames:
            return os.path.join(dirpath, frame_name)

    return None


def collect_frame_paths():
    frame_paths = []

    print("\nSearching for selected frames...")

    for frame_name in FRAME_NAMES:
        found_path = find_frame_file(frame_name)

        if found_path is None:
            print(f"Could not find frame: {frame_name}")
        else:
            print(f"Found frame: {frame_name}")
            print(f"  -> {found_path}")
            frame_paths.append(found_path)

    if len(frame_paths) == 0:
        print("\nNo selected frames were found.")
        print("Trying automatic fallback: search for any frame_*.jpg in the project...")

        fallback_frames = []

        for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
            for filename in filenames:
                lower_name = filename.lower()

                if lower_name.startswith("frame_") and lower_name.endswith((".jpg", ".jpeg", ".png")):
                    fallback_frames.append(os.path.join(dirpath, filename))

        fallback_frames = sorted(fallback_frames)

        if len(fallback_frames) == 0:
            raise FileNotFoundError(
                "No frame images were found anywhere in the project.\n"
                "Please check where your extracted frames are saved."
            )

        frame_paths = fallback_frames[:5]

        print("\nUsing these fallback frames:")
        for path in frame_paths:
            print("  ->", path)

    return frame_paths


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
# 4. LOAD OMNIDIRECTIONAL CALIBRATION
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

            # IMPORTANT:
            # OpenCV omnidir wants xi as a 1x1 CV_64F or CV_32F array.
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
        "\n".join(possible_files) +
        "\n\nFix: put your saved omnidirectional .npz file in results/ "
        "or add its exact path to possible_files."
    )


# ============================================================
# 5. DETECT CHARUCO
# ============================================================

def detect_charuco(image):
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
# 6. EQUIRECTANGULAR RECTIFICATION
# ============================================================

def make_equirectangular(image, K, D, xi):
    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64).reshape(-1, 1)
    xi = np.array(xi, dtype=np.float64).reshape(1, 1)

    h, w = image.shape[:2]

    # Larger output gives more pixels for visual inspection.
    # It does not completely solve seam splitting.
    out_w = 3 * w
    out_h = int(1.5 * h)

    R = np.eye(3, dtype=np.float64)

    P = np.array([
        [out_w / np.pi, 0, out_w / 2, 0],
        [0, out_h / np.pi, out_h / 2, 0],
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
        cv2.omnidir.RECTIFY_LONGLATI
    )

    equirectangular = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return equirectangular


# ============================================================
# 7. ROTATION FOR PERSPECTIVE VIEWS
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


# ============================================================
# 8. PERSPECTIVE RECTIFICATION
# ============================================================

def make_perspective(
    image,
    K,
    D,
    xi,
    yaw_deg=0,
    pitch_deg=0,
    focal_scale=0.35,
    canvas_scale=1.5
):
    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64).reshape(-1, 1)
    xi = np.array(xi, dtype=np.float64).reshape(1, 1)

    h, w = image.shape[:2]

    out_w = int(canvas_scale * w)
    out_h = int(canvas_scale * h)

    R = rotation_from_yaw_pitch(yaw_deg, pitch_deg)

    # Smaller focal_scale = wider view / less zoom.
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
# 9. MAIN DEBUG LOOP
# ============================================================

def main():
    print("Starting 5-frame equirectangular debug...")
    print("Project directory:", PROJECT_DIR)
    print("Output directory:", OUTPUT_DIR)

    frame_paths = collect_frame_paths()

    K, D, xi = load_omni_calibration()

    rows = []

    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            print(f"\nFrame not found, skipping: {frame_path}")
            continue

        image = cv2.imread(frame_path)

        if image is None:
            print(f"\nCould not read image, skipping: {frame_path}")
            continue

        frame_name = os.path.splitext(os.path.basename(frame_path))[0]

        print("\n======================================")
        print("Processing:", frame_name)
        print("Path:", frame_path)
        print("Image shape:", image.shape)
        print("======================================")

        # 1. Save raw image
        raw_out = os.path.join(
            OUTPUT_DIR,
            "raw",
            f"{frame_name}_raw.jpg"
        )
        cv2.imwrite(raw_out, image)

        # 2. Raw ChArUco detection
        raw_vis, raw_marker_count, raw_charuco_count = detect_charuco(image)

        raw_det_out = os.path.join(
            OUTPUT_DIR,
            "raw_detection",
            f"{frame_name}_raw_detection.jpg"
        )
        cv2.imwrite(raw_det_out, raw_vis)

        print("Raw markers:", raw_marker_count)
        print("Raw ChArUco corners:", raw_charuco_count)

        # 3. Equirectangular rectification
        equirectangular = make_equirectangular(image, K, D, xi)

        eq_out = os.path.join(
            OUTPUT_DIR,
            "equirectangular",
            f"{frame_name}_equirectangular.jpg"
        )
        cv2.imwrite(eq_out, equirectangular)

        # 4. Detection on equirectangular result
        eq_vis, eq_marker_count, eq_charuco_count = detect_charuco(equirectangular)

        eq_det_out = os.path.join(
            OUTPUT_DIR,
            "equirectangular_detection",
            f"{frame_name}_equirectangular_detection.jpg"
        )
        cv2.imwrite(eq_det_out, eq_vis)

        print("Equirectangular markers:", eq_marker_count)
        print("Equirectangular ChArUco corners:", eq_charuco_count)

        # 5. Best perspective rectification for detection
        best_perspective = make_perspective(
            image,
            K,
            D,
            xi,
            yaw_deg=0,
            pitch_deg=0,
            focal_scale=0.25,
            canvas_scale=1.5
        )

        best_persp_out = os.path.join(
            OUTPUT_DIR,
            "perspective_best",
            f"{frame_name}_perspective_best_focal_025.jpg"
        )
        cv2.imwrite(best_persp_out, best_perspective)

        # 6. ChArUco detection on best perspective image
        best_persp_vis, best_persp_marker_count, best_persp_charuco_count = detect_charuco(
            best_perspective
        )

        best_persp_det_out = os.path.join(
            OUTPUT_DIR,
            "perspective_best_detection",
            f"{frame_name}_perspective_best_focal_025_detection.jpg"
        )
        cv2.imwrite(best_persp_det_out, best_persp_vis)

        print("Best perspective markers:", best_persp_marker_count)
        print("Best perspective ChArUco corners:", best_persp_charuco_count)

        rows.append({
            "frame": frame_name,
            "source_path": frame_path,

            "raw_markers": raw_marker_count,
            "raw_charuco_corners": raw_charuco_count,

            "equirectangular_markers": eq_marker_count,
            "equirectangular_charuco_corners": eq_charuco_count,

            "best_perspective_markers": best_persp_marker_count,
            "best_perspective_charuco_corners": best_persp_charuco_count,

            "raw_image": raw_out,
            "raw_detection_image": raw_det_out,

            "equirectangular_image": eq_out,
            "equirectangular_detection_image": eq_det_out,

            "best_perspective_image": best_persp_out,
            "best_perspective_detection_image": best_persp_det_out,
        })

    summary = pd.DataFrame(rows)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n======================================")
    print("DONE")
    print("======================================")
    print("Summary saved to:", summary_path)

    if len(summary) == 0:
        print("No frames were processed.")
    else:
        print(summary)


if __name__ == "__main__":
    main()
