from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import load_useful_frame_paths
from .config import (
    DETECTION_SUMMARY_PATH,
    RESULTS_DIR,
    ensure_project_dirs,
)


OMNI_CALIBRATION_PATH = RESULTS_DIR / "omni_calibration.npz"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSPECTIVE_FRAMES_DIR = PROJECT_ROOT / "data" / "omni_perspective_frames"
PERSPECTIVE_SUMMARY_PATH = RESULTS_DIR / "omni_perspective_summary.csv"


@dataclass(frozen=True)
class PerspectiveResult:
    source_frame_path: Path
    perspective_frame_path: Path
    view_name: str
    width: int
    height: int
    error: str


def load_omni_calibration(
    calibration_path: Path = OMNI_CALIBRATION_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(calibration_path, allow_pickle=True)

    K = data["camera_matrix"].astype(np.float64)
    xi = data["xi"].astype(np.float64)
    D = data["dist_coeffs"].astype(np.float64)
    image_size = tuple(int(value) for value in data["image_size"])

    return K, xi, D, image_size


def rotation_matrix_from_euler(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """
    Create rotation matrix from yaw, pitch, roll angles in degrees.

    yaw   = left/right rotation
    pitch = up/down rotation
    roll  = image tilt
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Ryaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )

    Rpitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )

    Rroll = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Rroll @ Rpitch @ Ryaw


def build_perspective_maps(
    K: np.ndarray,
    xi: np.ndarray,
    D: np.ndarray,
    output_size: tuple[int, int],
    yaw_deg: float,
    pitch_deg: float,
    fov_scale: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a perspective rectification map from the omnidirectional model.
    """
    width, height = output_size

    focal = width * fov_scale

    P = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    R = rotation_matrix_from_euler(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
    )

    map_x, map_y = cv2.omnidir.initUndistortRectifyMap(
        K,
        D,
        xi,
        R,
        P,
        output_size,
        cv2.CV_32FC1,
        flags=cv2.omnidir.RECTIFY_PERSPECTIVE,
    )

    return map_x, map_y


def rectify_one_frame(
    source_frame_path: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    view_name: str,
) -> PerspectiveResult:
    PERSPECTIVE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PERSPECTIVE_FRAMES_DIR / source_frame_path.name.replace(
        ".jpg",
        f"_{view_name}.jpg",
    )

    frame = cv2.imread(str(source_frame_path))

    if frame is None:
        return PerspectiveResult(
            source_frame_path=source_frame_path,
            perspective_frame_path=output_path,
            view_name=view_name,
            width=0,
            height=0,
            error="Could not read frame",
        )

    perspective = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    cv2.imwrite(str(output_path), perspective)

    return PerspectiveResult(
        source_frame_path=source_frame_path,
        perspective_frame_path=output_path,
        view_name=view_name,
        width=perspective.shape[1],
        height=perspective.shape[0],
        error="",
    )


def write_summary(results: list[PerspectiveResult]) -> None:
    with PERSPECTIVE_SUMMARY_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_frame_path",
                "perspective_frame_path",
                "view_name",
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
                    "perspective_frame_path": result.perspective_frame_path,
                    "view_name": result.view_name,
                    "width": result.width,
                    "height": result.height,
                    "error": result.error,
                }
            )


def main() -> None:
    ensure_project_dirs()
    PERSPECTIVE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading omnidirectional calibration...")
    K, xi, D, image_size = load_omni_calibration()

    print(f"Original image size: {image_size}")
    print(f"K:\n{K}")
    print(f"xi:\n{xi}")
    print(f"D:\n{D.ravel()}")

    frame_paths = load_useful_frame_paths(DETECTION_SUMMARY_PATH)

    if not frame_paths:
        raise RuntimeError("No useful raw frames found. Run raw detection first.")

    output_size = (1280, 720)

    # We create several virtual camera views.
    # This helps because the board may appear in different regions of the omnidirectional image.
    views = [
        ("front", 0.0, 0.0),

        ("left_30", -30.0, 0.0),
        ("left_60", -60.0, 0.0),
        ("left_90", -90.0, 0.0),

        ("right_30", 30.0, 0.0),
        ("right_60", 60.0, 0.0),
        ("right_90", 90.0, 0.0),

        ("up_25", 0.0, -25.0),
        ("up_45", 0.0, -45.0),

        ("down_25", 0.0, 25.0),
        ("down_45", 0.0, 45.0),

        ("left_30_down_25", -30.0, 25.0),
        ("right_30_down_25", 30.0, 25.0),
        ("left_30_up_25", -30.0, -25.0),
        ("right_30_up_25", 30.0, -25.0),
    ]

    all_results: list[PerspectiveResult] = []

    for view_name, yaw_deg, pitch_deg in views:
        print(f"Building perspective view: {view_name}, yaw={yaw_deg}, pitch={pitch_deg}")

        map_x, map_y = build_perspective_maps(
            K=K,
            xi=xi,
            D=D,
            output_size=output_size,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_scale=0.75,
        )

        for frame_path in frame_paths:
            result = rectify_one_frame(
                source_frame_path=frame_path,
                map_x=map_x,
                map_y=map_y,
                view_name=view_name,
            )
            all_results.append(result)

    successful = [result for result in all_results if not result.error]

    write_summary(all_results)

    print(f"Perspective frames written: {len(successful)} / {len(all_results)}")
    print(f"Perspective frames directory: {PERSPECTIVE_FRAMES_DIR}")
    print(f"Perspective summary: {PERSPECTIVE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
