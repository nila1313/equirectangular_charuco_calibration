from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DETECTED_FRAMES_DIR,
    DETECTION_SUMMARY_PATH,
    RAW_FRAMES_DIR,
    BoardConfig,
    RawDetectionConfig,
    find_default_video,
)
from .detector import DetectionResult, run_raw_detection


def parse_args() -> RawDetectionConfig:
    parser = argparse.ArgumentParser(
        description="Extract video frames and detect raw ChArUco corners."
    )
    parser.add_argument("--video", type=Path, default=None, help="Path to calibration video.")
    parser.add_argument("--frame-step", type=int, default=RawDetectionConfig.frame_step, help="Sample every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N sampled frames.")
    parser.add_argument("--dictionary", default=BoardConfig.dictionary_name, help="OpenCV ArUco dictionary.")
    parser.add_argument("--squares-x", type=int, default=BoardConfig.squares_x, help="ChArUco board squares in X.")
    parser.add_argument("--squares-y", type=int, default=BoardConfig.squares_y, help="ChArUco board squares in Y.")
    parser.add_argument("--square-length", type=float, default=BoardConfig.square_length, help="Square side length.")
    parser.add_argument("--marker-length", type=float, default=BoardConfig.marker_length, help="Marker side length.")
    parser.add_argument(
        "--min-markers",
        type=int,
        default=6,
        help="Minimum detected markers to flag a frame as useful.",
    )
    args = parser.parse_args()

    video_path = args.video if args.video is not None else find_default_video()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    board = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
    )

    return RawDetectionConfig(
        video_path=video_path,
        board=board,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_markers=args.min_markers,
    )


def print_summary(config: RawDetectionConfig, results: list[DetectionResult]) -> None:
    detected = [result for result in results if result.marker_count > 0]
    useful = [result for result in results if result.used_for_calibration]
    print(f"Video input: {config.video_path}")
    print(f"Sampled frames: {len(results)}")
    print(f"Frames with markers: {len(detected)}")
    print(f"Frames marked useful: {len(useful)}")
    print(f"Raw frames: {RAW_FRAMES_DIR}")
    print(f"Detected frames: {DETECTED_FRAMES_DIR}")
    print(f"Detection summary: {DETECTION_SUMMARY_PATH}")


def main() -> None:
    config = parse_args()
    results = run_raw_detection(config)
    print_summary(config, results)


if __name__ == "__main__":
    main()
