from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import load_useful_frame_paths
from .config import (
    DETECTION_SUMMARY_PATH,
    RECTIFIED_FRAMES_DIR,
    RESULTS_DIR,
    ensure_project_dirs,
)


OMNI_CALIBRATION_PATH = RESULTS_DIR / "omni_calibration.npz"
EQUIRECTANGULAR_FRAMES_DIR = Path(__file__).resolve().parents[1] / "data" / "equirectangular_frames"
EQUIRECTANGULAR_SUMMARY_PATH = RESULTS_DIR / "equirectangular_summary.csv"


@dataclass(frozen=True)
class EquirectangularResult:
    source_frame_path: Path
    equirectangular_frame_path: Path
    width: int
    height: int
    error: str


def load_omni_calibration(
    calibration_path: Path = OMNI_CALIBRATION_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Load omnidirectional calibration parameters:
    K, xi, D, and image size.
    """
    data = np.load(calibration_path, allow_pickle=True)

    K = data["camera_matrix"].astype(np.float64)
    xi = data["xi"].astype(np.float64)
    D = data["dist_coeffs"].astype(np.float64)
    image_size = tuple(int(value) for value in data["image_size"])

    return K, xi, D, image_size


def build_equirectangular_maps(
    K: np.ndarray,
    xi: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Build longitude-latitude / equirectangular rectification maps.

    OpenCV calls this RECTIFY_LONGLATI.
    This is the equirectangular-like projection.
    """
    width, height = image_size

    # Output size.
    # We keep the same size first to make comparison easy.
    output_size = (width, height)

    # R is identity because we do not rotate the virtual view yet.
    R = np.eye(3, dtype=np.float64)

    # P controls the virtual camera/projection.
    # For RECTIFY_LONGLATI, OpenCV still expects a 3x3 matrix.
    # We start with a simple matrix based on the original image size.
    P = np.array(
        [
            [width / np.pi, 0.0, width / 2.0],
            [0.0, height / np.pi, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    map_x, map_y = cv2.omnidir.initUndistortRectifyMap(
        K,
        D,
        xi,
        R,
        P,
        output_size,
        cv2.CV_32FC1,
        flags=cv2.omnidir.RECTIFY_LONGLATI,
    )

    return map_x, map_y, output_size


def equirectangular_rectify_frame(
    source_frame_path: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> EquirectangularResult:
    """
    Apply equirectangular rectification to one raw frame.
    """
    EQUIRECTANGULAR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EQUIRECTANGULAR_FRAMES_DIR / source_frame_path.name.replace(
        ".jpg",
        "_equirectangular.jpg",
    )

    frame = cv2.imread(str(source_frame_path))

    if frame is None:
        return EquirectangularResult(
            source_frame_path=source_frame_path,
            equirectangular_frame_path=output_path,
            width=0,
            height=0,
            error="Could not read frame",
        )

    equirectangular = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    cv2.imwrite(str(output_path), equirectangular)

    return EquirectangularResult(
        source_frame_path=source_frame_path,
        equirectangular_frame_path=output_path,
        width=equirectangular.shape[1],
        height=equirectangular.shape[0],
        error="",
    )


def write_equirectangular_summary(
    results: list[EquirectangularResult],
    output_path: Path = EQUIRECTANGULAR_SUMMARY_PATH,
) -> None:
    """
    Write CSV summary for generated equirectangular frames.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_frame_path",
                "equirectangular_frame_path",
                "width",
                "height",
                "error",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "source_frame_path": result.source_frame_path,
                    "equirectangular_frame_path": result.equirectangular_frame_path,
                    "width": result.width,
                    "height": result.height,
                    "error": result.error,
                }
            )


def main() -> None:
    ensure_project_dirs()
    EQUIRECTANGULAR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading omnidirectional calibration...")
    K, xi, D, image_size = load_omni_calibration()

    print(f"Image size: {image_size}")
    print(f"K:\n{K}")
    print(f"xi:\n{xi}")
    print(f"D:\n{D.ravel()}")

    print("Building equirectangular maps...")
    map_x, map_y, output_size = build_equirectangular_maps(
        K=K,
        xi=xi,
        D=D,
        image_size=image_size,
    )

    print(f"Equirectangular output size: {output_size}")

    frame_paths = load_useful_frame_paths(DETECTION_SUMMARY_PATH)

    if not frame_paths:
        raise RuntimeError("No useful raw frames found. Run raw detection first.")

    print(f"Frames to equirectangular-rectify: {len(frame_paths)}")

    results = [
        equirectangular_rectify_frame(frame_path, map_x, map_y)
        for frame_path in frame_paths
    ]

    successful = [result for result in results if not result.error]

    write_equirectangular_summary(results)

    print(f"Equirectangular frames written: {len(successful)} / {len(results)}")
    print(f"Equirectangular frames directory: {EQUIRECTANGULAR_FRAMES_DIR}")
    print(f"Equirectangular summary: {EQUIRECTANGULAR_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
    