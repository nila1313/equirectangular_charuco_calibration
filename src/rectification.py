from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import load_useful_frame_paths
from .config import (
    DETECTION_SUMMARY_PATH,
    PRELIMINARY_CALIBRATION_PATH,
    RECTIFICATION_SUMMARY_PATH,
    RECTIFIED_FRAMES_DIR,
    ensure_project_dirs,
)


@dataclass(frozen=True)
class RectificationConfig:
    calibration_path: Path = PRELIMINARY_CALIBRATION_PATH
    detection_summary_path: Path = DETECTION_SUMMARY_PATH
    output_summary_path: Path = RECTIFICATION_SUMMARY_PATH
    alpha: float = 0.0
    crop_valid_roi: bool = True


@dataclass(frozen=True)
class RectificationResult:
    source_frame_path: Path
    rectified_frame_path: Path
    width: int
    height: int
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    error: str


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(calibration_path, allow_pickle=True)
    image_size = tuple(int(value) for value in data["image_size"])
    return data["camera_matrix"], data["dist_coeffs"], image_size


def build_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )
    return map_x, map_y, roi


def rectify_frame(
    source_frame_path: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    roi: tuple[int, int, int, int],
    crop_valid_roi: bool,
) -> RectificationResult:
    output_path = RECTIFIED_FRAMES_DIR / source_frame_path.name.replace(
        ".jpg",
        "_rectified.jpg",
    )
    frame = cv2.imread(str(source_frame_path))
    if frame is None:
        return RectificationResult(source_frame_path, output_path, 0, 0, 0, 0, 0, 0, "Could not read frame")

    rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    x, y, width, height = roi
    if crop_valid_roi and width > 0 and height > 0:
        rectified = rectified[y : y + height, x : x + width]

    cv2.imwrite(str(output_path), rectified)
    return RectificationResult(
        source_frame_path=source_frame_path,
        rectified_frame_path=output_path,
        width=rectified.shape[1],
        height=rectified.shape[0],
        roi_x=x,
        roi_y=y,
        roi_width=width,
        roi_height=height,
        error="",
    )


def write_rectification_summary(results: list[RectificationResult], output_path: Path) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_frame_path",
                "rectified_frame_path",
                "width",
                "height",
                "roi_x",
                "roi_y",
                "roi_width",
                "roi_height",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_frame_path": result.source_frame_path,
                    "rectified_frame_path": result.rectified_frame_path,
                    "width": result.width,
                    "height": result.height,
                    "roi_x": result.roi_x,
                    "roi_y": result.roi_y,
                    "roi_width": result.roi_width,
                    "roi_height": result.roi_height,
                    "error": result.error,
                }
            )


def run_rectification(config: RectificationConfig) -> list[RectificationResult]:
    ensure_project_dirs()
    camera_matrix, dist_coeffs, image_size = load_calibration(config.calibration_path)
    map_x, map_y, roi = build_undistort_maps(
        camera_matrix,
        dist_coeffs,
        image_size,
        config.alpha,
    )

    frame_paths = load_useful_frame_paths(config.detection_summary_path)
    if not frame_paths:
        raise RuntimeError(f"No useful frames found in {config.detection_summary_path}")

    results = [
        rectify_frame(frame_path, map_x, map_y, roi, config.crop_valid_roi)
        for frame_path in frame_paths
    ]
    write_rectification_summary(results, config.output_summary_path)
    return results


def parse_args() -> RectificationConfig:
    parser = argparse.ArgumentParser(
        description="Rectify useful raw frames using the preliminary calibration."
    )
    parser.add_argument("--calibration", type=Path, default=PRELIMINARY_CALIBRATION_PATH)
    parser.add_argument("--summary", type=Path, default=DETECTION_SUMMARY_PATH)
    parser.add_argument("--output-summary", type=Path, default=RECTIFICATION_SUMMARY_PATH)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="0 crops black borders, 1 keeps all pixels.",
    )
    parser.add_argument(
        "--keep-full-frame",
        action="store_true",
        help="Do not crop to OpenCV's valid undistorted ROI.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    return RectificationConfig(
        calibration_path=args.calibration,
        detection_summary_path=args.summary,
        output_summary_path=args.output_summary,
        alpha=args.alpha,
        crop_valid_roi=not args.keep_full_frame,
    )


def main() -> None:
    config = parse_args()
    results = run_rectification(config)
    successful = [result for result in results if not result.error]
    print(f"Rectified frames: {len(successful)} / {len(results)}")
    print(f"Rectified frames directory: {RECTIFIED_FRAMES_DIR}")
    print(f"Rectification summary: {config.output_summary_path}")


if __name__ == "__main__":
    main()
