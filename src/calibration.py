from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DETECTION_SUMMARY_PATH,
    PRELIMINARY_CALIBRATION_PATH,
    BoardConfig,
    CalibrationConfig,
    ensure_project_dirs,
)
from .detector import CharucoDetector


def load_useful_frame_paths(
    summary_path: Path,
    frame_path_column: str = "raw_frame_path",
) -> list[Path]:
    with summary_path.open(newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        return [
            Path(row[frame_path_column])
            for row in rows
            if row["used_for_calibration"] == "True"
        ]


def run_preliminary_calibration(
    config: CalibrationConfig,
    summary_path: Path = DETECTION_SUMMARY_PATH,
    frame_path_column: str = "raw_frame_path",
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], int]:
    ensure_project_dirs()
    frame_paths = load_useful_frame_paths(summary_path, frame_path_column)
    if not frame_paths:
        raise RuntimeError(f"No useful frames found in {summary_path}")

    detector = CharucoDetector(config.board)
    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        image_size = (frame.shape[1], frame.shape[0])
        detection = detector.detect(frame)
        if (
            detection.charuco_corners is not None
            and detection.charuco_ids is not None
            and detection.charuco_corner_count >= config.min_corners
        ):
            all_charuco_corners.append(detection.charuco_corners)
            all_charuco_ids.append(detection.charuco_ids)

    if image_size is None:
        raise RuntimeError("Could not read any useful frame images.")
    if len(all_charuco_corners) < 3:
        raise RuntimeError(
            "Need at least 3 useful ChArUco frames for preliminary calibration."
        )

    width, height = image_size
    initial_focal_length = max(width, height) * 0.8
    initial_camera_matrix = np.array(
        [
            [initial_focal_length, 0.0, width / 2.0],
            [0.0, initial_focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    initial_dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        detector.board,
        image_size,
        initial_camera_matrix,
        initial_dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    np.savez(
        config.output_path,
        rms=rms,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        image_size=np.array(image_size),
        frame_count=len(all_charuco_corners),
        squares_x=config.board.squares_x,
        squares_y=config.board.squares_y,
        square_length=config.board.square_length,
        marker_length=config.board.marker_length,
        dictionary_name=config.board.dictionary_name,
    )

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs, len(all_charuco_corners)


def parse_args() -> tuple[CalibrationConfig, Path]:
    parser = argparse.ArgumentParser(
        description="Run preliminary pinhole calibration from useful raw ChArUco frames."
    )
    parser.add_argument("--summary", type=Path, default=DETECTION_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=PRELIMINARY_CALIBRATION_PATH)
    parser.add_argument("--dictionary", default=BoardConfig.dictionary_name)
    parser.add_argument("--squares-x", type=int, default=BoardConfig.squares_x)
    parser.add_argument("--squares-y", type=int, default=BoardConfig.squares_y)
    parser.add_argument("--square-length", type=float, default=BoardConfig.square_length)
    parser.add_argument("--marker-length", type=float, default=BoardConfig.marker_length)
    parser.add_argument("--min-corners", type=int, default=CalibrationConfig.min_corners)
    args = parser.parse_args()

    board = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
    )
    return CalibrationConfig(board=board, min_corners=args.min_corners, output_path=args.output), args.summary


def main() -> None:
    config, summary_path = parse_args()
    rms, camera_matrix, dist_coeffs, _, _, frame_count = run_preliminary_calibration(
        config,
        summary_path,
    )

    print(f"Calibrated frames: {frame_count}")
    print(f"RMS reprojection error: {rms:.6f}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
    print(f"Saved preliminary calibration: {config.output_path}")


if __name__ == "__main__":
    main()
