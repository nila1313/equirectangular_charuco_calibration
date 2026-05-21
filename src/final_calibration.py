from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import run_preliminary_calibration
from .config import (
    FINAL_CALIBRATION_PATH,
    RECTIFIED_DETECTION_SUMMARY_PATH,
    BoardConfig,
    CalibrationConfig,
)


def parse_args() -> tuple[CalibrationConfig, Path]:
    parser = argparse.ArgumentParser(
        description="Run final calibration from rectified ChArUco detections."
    )
    parser.add_argument("--summary", type=Path, default=RECTIFIED_DETECTION_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=FINAL_CALIBRATION_PATH)
    parser.add_argument("--dictionary", default=BoardConfig.dictionary_name)
    parser.add_argument("--squares-x", type=int, default=BoardConfig.squares_x)
    parser.add_argument("--squares-y", type=int, default=BoardConfig.squares_y)
    parser.add_argument("--square-length", type=float, default=BoardConfig.square_length)
    parser.add_argument("--marker-length", type=float, default=BoardConfig.marker_length)
    parser.add_argument("--min-corners", type=int, default=6)
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
        frame_path_column="rectified_frame_path",
    )

    print(f"Final calibrated frames: {frame_count}")
    print(f"RMS reprojection error: {rms:.6f}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
    print(f"Saved final calibration: {config.output_path}")


if __name__ == "__main__":
    main()
