from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from .config import (
    RECTIFICATION_SUMMARY_PATH,
    RECTIFIED_DETECTED_FRAMES_DIR,
    RECTIFIED_DETECTION_SUMMARY_PATH,
    RECTIFIED_FRAMES_DIR,
    BoardConfig,
    ensure_project_dirs,
)
from .detector import CharucoDetector, DetectionResult


def find_rectified_frames(rectified_frames_dir: Path) -> list[Path]:
    return sorted(rectified_frames_dir.glob("*_rectified.jpg"))


def load_rectified_frame_paths(rectification_summary_path: Path) -> list[Path]:
    with rectification_summary_path.open(newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        return [
            Path(row["rectified_frame_path"])
            for row in rows
            if not row["error"]
        ]


def detect_rectified_frame(
    frame_path: Path,
    detector: CharucoDetector,
    min_markers: int,
) -> DetectionResult:
    detected_frame_path = RECTIFIED_DETECTED_FRAMES_DIR / frame_path.name.replace(
        "_rectified.jpg",
        "_rectified_detected.jpg",
    )
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return DetectionResult(
            frame_index=-1,
            raw_frame_path=frame_path,
            detected_frame_path=None,
            marker_count=0,
            charuco_corner_count=0,
            marker_ids="",
            used_for_calibration=False,
            error="Could not read frame",
        )

    detection = detector.detect(frame)
    if detection.marker_count > 0:
        annotated = detector.draw(frame, detection)
        cv2.imwrite(str(detected_frame_path), annotated)
    else:
        detected_frame_path = None

    return DetectionResult(
        frame_index=-1,
        raw_frame_path=frame_path,
        detected_frame_path=detected_frame_path,
        marker_count=detection.marker_count,
        charuco_corner_count=detection.charuco_corner_count,
        marker_ids=detection.marker_ids_text,
        used_for_calibration=(
            detection.marker_count >= min_markers
            and detection.charuco_corner_count >= min_markers
        ),
        error=detection.error,
    )


def write_rectified_detection_summary(
    results: list[DetectionResult],
    summary_path: Path,
) -> None:
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rectified_frame_path",
                "detected_frame_path",
                "marker_count",
                "charuco_corner_count",
                "marker_ids",
                "used_for_calibration",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "rectified_frame_path": result.raw_frame_path,
                    "detected_frame_path": result.detected_frame_path or "",
                    "marker_count": result.marker_count,
                    "charuco_corner_count": result.charuco_corner_count,
                    "marker_ids": result.marker_ids,
                    "used_for_calibration": result.used_for_calibration,
                    "error": result.error,
                }
            )


def run_rectified_detection(
    board_config: BoardConfig,
    min_markers: int,
    rectified_frames_dir: Path = RECTIFIED_FRAMES_DIR,
    summary_path: Path = RECTIFIED_DETECTION_SUMMARY_PATH,
    rectification_summary_path: Path = RECTIFICATION_SUMMARY_PATH,
) -> list[DetectionResult]:
    ensure_project_dirs()
    if rectification_summary_path.exists():
        frame_paths = load_rectified_frame_paths(rectification_summary_path)
    else:
        frame_paths = find_rectified_frames(rectified_frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No rectified frames found in {rectified_frames_dir}")

    detector = CharucoDetector(board_config)
    results = [
        detect_rectified_frame(frame_path, detector, min_markers)
        for frame_path in frame_paths
    ]
    write_rectified_detection_summary(results, summary_path)
    return results


def parse_args() -> tuple[BoardConfig, int, Path, Path, Path]:
    parser = argparse.ArgumentParser(
        description="Detect ChArUco corners on rectified frames."
    )
    parser.add_argument("--rectified-frames-dir", type=Path, default=RECTIFIED_FRAMES_DIR)
    parser.add_argument("--summary", type=Path, default=RECTIFIED_DETECTION_SUMMARY_PATH)
    parser.add_argument("--rectification-summary", type=Path, default=RECTIFICATION_SUMMARY_PATH)
    parser.add_argument("--dictionary", default=BoardConfig.dictionary_name)
    parser.add_argument("--squares-x", type=int, default=BoardConfig.squares_x)
    parser.add_argument("--squares-y", type=int, default=BoardConfig.squares_y)
    parser.add_argument("--square-length", type=float, default=BoardConfig.square_length)
    parser.add_argument("--marker-length", type=float, default=BoardConfig.marker_length)
    parser.add_argument("--min-markers", type=int, default=6)
    args = parser.parse_args()

    board = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
    )
    return board, args.min_markers, args.rectified_frames_dir, args.summary, args.rectification_summary


def main() -> None:
    board, min_markers, rectified_frames_dir, summary_path, rectification_summary_path = parse_args()
    results = run_rectified_detection(
        board,
        min_markers,
        rectified_frames_dir,
        summary_path,
        rectification_summary_path,
    )
    detected = [result for result in results if result.marker_count > 0]
    useful = [result for result in results if result.used_for_calibration]
    print(f"Rectified frames checked: {len(results)}")
    print(f"Rectified frames with markers: {len(detected)}")
    print(f"Rectified frames useful for final calibration: {len(useful)}")
    print(f"Rectified detections: {RECTIFIED_DETECTED_FRAMES_DIR}")
    print(f"Rectified detection summary: {summary_path}")


if __name__ == "__main__":
    main()
