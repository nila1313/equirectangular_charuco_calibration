from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .config import (
    BoardConfig,
    CalibrationConfig,
    DETECTION_SUMMARY_PATH,
    RESULTS_DIR,
    ensure_project_dirs,
)
from .detector import CharucoDetector


OMNI_CALIBRATION_PATH = RESULTS_DIR / "omni_calibration.npz"


def load_useful_raw_frame_paths(
    summary_path: Path = DETECTION_SUMMARY_PATH,
) -> list[Path]:
    """
    Read detection_summary.csv and return raw frame paths marked useful
    during the first ChArUco detection pass.
    """
    with summary_path.open(newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        return [
            Path(row["raw_frame_path"])
            for row in rows
            if row["used_for_calibration"] == "True"
        ]


def collect_charuco_points(
    frame_paths: list[Path],
    board_config: BoardConfig,
    min_corners: int,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """
    Detect ChArUco corners again on useful raw frames.

    For each frame we create:
    - object points: real 3D ChArUco board corner coordinates
    - image points: detected 2D pixel coordinates

    cv2.omnidir.calibrate needs these matching 3D-2D correspondences.
    """
    detector = CharucoDetector(board_config)
    board_corners = detector.board.getChessboardCorners()

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    image_size: tuple[int, int] | None = None

    skipped_unreadable = 0
    skipped_no_detection = 0
    skipped_too_few_corners = 0

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))

        if frame is None:
            skipped_unreadable += 1
            continue

        image_size = (frame.shape[1], frame.shape[0])

        detection = detector.detect(frame)

        if detection.charuco_corners is None or detection.charuco_ids is None:
            skipped_no_detection += 1
            continue

        if detection.charuco_corner_count < min_corners:
            skipped_too_few_corners += 1
            continue

        obj = []
        img = []

        for corner, corner_id in zip(
            detection.charuco_corners,
            detection.charuco_ids.flatten(),
        ):
            obj.append(board_corners[int(corner_id)])
            img.append(corner[0])

        # OpenCV omnidir accepts each frame as Mat with shape 1xN or Nx1.
        # We use 1xN with 3 channels for object points and 2 channels for image points.
        obj = np.asarray(obj, dtype=np.float64).reshape(1, -1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(1, -1, 2)

        object_points.append(obj)
        image_points.append(img)

    if image_size is None:
        raise RuntimeError("Could not read any frame.")

    print(f"Skipped unreadable frames: {skipped_unreadable}")
    print(f"Skipped frames without ChArUco detection: {skipped_no_detection}")
    print(f"Skipped frames with too few corners: {skipped_too_few_corners}")

    return object_points, image_points, image_size


def make_initial_camera_matrix(image_size: tuple[int, int]) -> np.ndarray:
    """
    Create an initial camera matrix guess.

    K =
    [ fx  0  cx ]
    [  0 fy  cy ]
    [  0  0   1 ]

    fx and fy are initialized from the image size.
    cx and cy are initialized at the image center.
    """
    width, height = image_size
    initial_focal_length = max(width, height) * 0.8

    return np.array(
        [
            [initial_focal_length, 0.0, width / 2.0],
            [0.0, initial_focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def run_omnidir_calibration(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    """
    Run OpenCV omnidirectional calibration.

    This estimates:
    - K: camera matrix
    - xi: omnidirectional mirror/sphere model parameter
    - D: distortion coefficients k1, k2, p1, p2
    - rvecs/tvecs: board pose for each used image
    """
    K = make_initial_camera_matrix(image_size)

    # xi is the extra parameter in Mei's omnidirectional camera model.
    xi = np.zeros((1, 1), dtype=np.float64)

    # OpenCV omnidir distortion has 4 parameters: k1, k2, p1, p2.
    D = np.zeros((1, 4), dtype=np.float64)

    flags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        200,
        1e-8,
    )

    print("Running cv2.omnidir.calibrate...")
    print(f"Number of frames passed to omnidir: {len(object_points)}")
    print(f"First object point shape: {object_points[0].shape}")
    print(f"First image point shape: {image_points[0].shape}")
    print(f"Image size: {image_size}")

    rms, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(
        object_points,
        image_points,
        image_size,
        K,
        xi,
        D,
        flags,
        criteria,
    )

    return rms, K, xi, D, rvecs, tvecs, idx


def main() -> None:
    ensure_project_dirs()

    frame_paths = load_useful_raw_frame_paths()
    print(f"Useful raw frames found: {len(frame_paths)}")

    if not frame_paths:
        raise RuntimeError("No useful raw frames found. Run raw detection first.")

    object_points, image_points, image_size = collect_charuco_points(
        frame_paths=frame_paths,
        board_config=BoardConfig(),
        min_corners=CalibrationConfig.min_corners,
    )

    print(f"Frames usable for omnidir calibration: {len(object_points)}")
    print(f"Image size: {image_size}")

    if len(object_points) < 3:
        raise RuntimeError("Need at least 3 frames for omnidirectional calibration.")

    rms, K, xi, D, rvecs, tvecs, idx = run_omnidir_calibration(
        object_points,
        image_points,
        image_size,
    )

    np.savez(
        OMNI_CALIBRATION_PATH,
        rms=rms,
        camera_matrix=K,
        xi=xi,
        dist_coeffs=D,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        idx=idx,
        image_size=np.array(image_size),
        frame_count=len(object_points),
    )

    print("")
    print("Omnidirectional calibration finished.")
    print(f"Omnidirectional calibrated frames: {len(object_points)}")
    print(f"Omnidirectional RMS reprojection error: {rms:.6f}")
    print(f"Camera matrix:\n{K}")
    print(f"Xi:\n{xi}")
    print(f"Distortion coefficients:\n{D.ravel()}")
    print(f"Used frame indices from OpenCV idx:\n{idx}")
    print(f"Saved omnidirectional calibration: {OMNI_CALIBRATION_PATH}")


if __name__ == "__main__":
    main()