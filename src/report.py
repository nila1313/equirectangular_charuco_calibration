from __future__ import annotations

import argparse
import csv
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .config import (
    DETECTED_FRAMES_DIR,
    DETECTION_SUMMARY_PATH,
    FINAL_CALIBRATION_PATH,
    PRELIMINARY_CALIBRATION_PATH,
    RECTIFIED_DETECTED_FRAMES_DIR,
    RECTIFIED_DETECTION_SUMMARY_PATH,
    RESULTS_DIR,
)


REPORT_PATH = RESULTS_DIR / "calibration_report.pdf"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def count_rows_with_markers(rows: list[dict[str, str]]) -> int:
    return sum(int(row["marker_count"]) > 0 for row in rows)


def count_useful_rows(rows: list[dict[str, str]]) -> int:
    return sum(row["used_for_calibration"] == "True" for row in rows)


def average_corners(rows: list[dict[str, str]]) -> float:
    useful = [row for row in rows if row["used_for_calibration"] == "True"]
    if not useful:
        return 0.0
    return sum(int(row["charuco_corner_count"]) for row in useful) / len(useful)


def add_wrapped_text(
    ax: plt.Axes,
    text: str,
    x: float = 0.08,
    y: float = 0.92,
    size: int = 11,
    width: int = 88,
    line_spacing: float = 0.055,
) -> None:
    ax.axis("off")
    current_y = y
    for paragraph in text.split("\n"):
        if not paragraph:
            current_y -= line_spacing
            continue
        for line in textwrap.wrap(paragraph, width=width):
            ax.text(x, current_y, line, fontsize=size, va="top", family="DejaVu Sans")
            current_y -= line_spacing


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.axis("off")
    ax.text(0.08, 0.88, title, fontsize=24, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.08, 0.78, subtitle, fontsize=12, va="top")


def matrix_text(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=6, suppress_small=False)


def pick_example_image(directory: Path) -> Path | None:
    images = sorted(path for path in directory.glob("*.jpg") if path.is_file())
    if not images:
        return None
    return images[len(images) // 2]


def add_image_page(pdf: PdfPages, title: str, image_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.08, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if image_path is None:
        ax.text(0.08, 0.86, "No example image found.", fontsize=12, va="top")
    else:
        image = mpimg.imread(image_path)
        ax.imshow(image, extent=(0.08, 0.92, 0.20, 0.82), aspect="auto")
        ax.text(0.08, 0.16, str(image_path), fontsize=8, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_report(output_path: Path) -> None:
    raw_rows = load_csv(DETECTION_SUMMARY_PATH)
    rectified_rows = load_csv(RECTIFIED_DETECTION_SUMMARY_PATH)
    preliminary = load_calibration(PRELIMINARY_CALIBRATION_PATH)
    final = load_calibration(FINAL_CALIBRATION_PATH)

    raw_markers = count_rows_with_markers(raw_rows)
    raw_useful = count_useful_rows(raw_rows)
    rectified_markers = count_rows_with_markers(rectified_rows)
    rectified_useful = count_useful_rows(rectified_rows)

    preliminary_rms = float(preliminary["rms"])
    final_rms = float(final["rms"])
    improvement = (preliminary_rms - final_rms) / preliminary_rms * 100.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        add_title(
            ax,
            "Equirectangular ChArUco Calibration Report",
            "Step-by-step summary of the current calibration pipeline.",
        )
        add_wrapped_text(
            ax,
            "\n".join(
                [
                    "Goal: calibrate an omnidirectional / fisheye camera using a ChArUco board.",
                    "",
                    "Pipeline:",
                    "1. Sample raw frames from the calibration video.",
                    "2. Detect ChArUco markers and corners in raw frames.",
                    "3. Use high-quality raw detections for preliminary calibration.",
                    "4. Rectify useful raw frames with the preliminary calibration.",
                    "5. Detect ChArUco corners again on rectified frames.",
                    "6. Use the cleaner rectified detections for final calibration.",
                    "",
                    f"Final RMS reprojection error: {final_rms:.6f}",
                    f"Final calibrated frames: {int(final['frame_count'])}",
                ]
            ),
            y=0.66,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        add_wrapped_text(
            ax,
            "\n".join(
                [
                    "Step 1 - Raw frame detection",
                    "",
                    f"Input video frames were sampled every 30 frames.",
                    f"Sampled frames: {len(raw_rows)}",
                    f"Frames with markers: {raw_markers}",
                    f"Frames marked useful: {raw_useful}",
                    f"Average ChArUco corners in useful raw frames: {average_corners(raw_rows):.2f}",
                    "",
                    "Outputs:",
                    "data/raw_frames/",
                    "data/detected_frames/",
                    "results/detection_summary.csv",
                ]
            ),
            size=12,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        add_wrapped_text(
            ax,
            "\n".join(
                [
                    "Step 2 - Preliminary calibration",
                    "",
                    "The preliminary calibration uses stricter raw detections, requiring at least 9 ChArUco corners per frame.",
                    f"Frames used: {int(preliminary['frame_count'])}",
                    f"RMS reprojection error: {preliminary_rms:.6f}",
                    "",
                    "Camera matrix:",
                    matrix_text(preliminary["camera_matrix"]),
                    "",
                    "Distortion coefficients:",
                    matrix_text(preliminary["dist_coeffs"].ravel()),
                    "",
                    "Output:",
                    "results/preliminary_calibration.npz",
                ]
            ),
            size=10,
            width=96,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        add_wrapped_text(
            ax,
            "\n".join(
                [
                    "Step 3 - Rectification and second-pass detection",
                    "",
                    "Useful raw frames are rectified using the preliminary calibration. Full-frame rectification is used with alpha=1 to preserve more board area.",
                    f"Rectified frames checked: {len(rectified_rows)}",
                    f"Rectified frames with markers: {rectified_markers}",
                    f"Rectified frames useful for final calibration: {rectified_useful}",
                    f"Average ChArUco corners in useful rectified frames: {average_corners(rectified_rows):.2f}",
                    "",
                    "Outputs:",
                    "data/rectified_frames/",
                    "data/rectified_detected_frames/",
                    "results/rectification_summary.csv",
                    "results/rectified_detection_summary.csv",
                ]
            ),
            size=12,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        add_wrapped_text(
            ax,
            "\n".join(
                [
                    "Step 4 - Final calibration",
                    "",
                    "The final calibration uses rectified detections. This pass uses at least 6 ChArUco corners per frame because detections are cleaner after rectification.",
                    f"Frames used: {int(final['frame_count'])}",
                    f"RMS reprojection error: {final_rms:.6f}",
                    f"RMS improvement from preliminary to final: {improvement:.2f}%",
                    "",
                    "Camera matrix:",
                    matrix_text(final["camera_matrix"]),
                    "",
                    "Distortion coefficients:",
                    matrix_text(final["dist_coeffs"].ravel()),
                    "",
                    "Output:",
                    "results/final_calibration.npz",
                ]
            ),
            size=10,
            width=96,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        labels = ["Raw useful", "Rectified useful", "Prelim frames", "Final frames"]
        values = [
            raw_useful,
            rectified_useful,
            int(preliminary["frame_count"]),
            int(final["frame_count"]),
        ]
        ax.bar(labels, values, color=["#3b82f6", "#16a34a", "#6366f1", "#0891b2"])
        ax.set_title("Frame Counts by Stage")
        ax.set_ylabel("Frame count")
        ax.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_image_page(pdf, "Example Raw ChArUco Detection", pick_example_image(DETECTED_FRAMES_DIR))
        add_image_page(
            pdf,
            "Example Rectified ChArUco Detection",
            pick_example_image(RECTIFIED_DETECTED_FRAMES_DIR),
        )


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Generate calibration PDF report.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    return parser.parse_args().output


def main() -> None:
    output_path = parse_args()
    make_report(output_path)
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
