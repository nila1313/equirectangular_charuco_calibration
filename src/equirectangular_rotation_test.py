from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import RESULTS_DIR, DETECTION_SUMMARY_PATH
from .calibration import load_useful_frame_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OMNI_CALIBRATION_PATH = RESULTS_DIR / "omni_calibration.npz"
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "equirectangular_rotation_test"


def load_omni_calibration():
    data = np.load(OMNI_CALIBRATION_PATH, allow_pickle=True)

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


def build_lONGLATI_maps(
    K: np.ndarray,
    xi: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    width, height = image_size

    # 2:1-like wide longitude-latitude canvas.
    output_size = (2 * width, height)

    R = rotation_matrix_from_euler(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
    )

    # Full longitude-latitude scale.
    P = np.array(
        [
            [output_size[0] / (2.0 * np.pi), 0.0, output_size[0] / 2.0],
            [0.0, output_size[1] / np.pi, output_size[1] / 2.0],
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


def main() -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    K, xi, D, image_size = load_omni_calibration()

    frame_paths = load_useful_frame_paths(DETECTION_SUMMARY_PATH)

    if not frame_paths:
        raise RuntimeError("No useful raw frames found.")

    # Use the first useful frame for testing.
    # Later we can change this to another frame if needed.
    source_frame_path = frame_paths[0]
    frame = cv2.imread(str(source_frame_path))

    if frame is None:
        raise RuntimeError(f"Could not read frame: {source_frame_path}")

    print(f"Testing frame: {source_frame_path}")
    print(f"Image size: {image_size}")

    yaw_values = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    pitch_values = [-45, -25, 0, 25, 45]

    count = 0

    for yaw in yaw_values:
        for pitch in pitch_values:
            map_x, map_y, output_size = build_lONGLATI_maps(
                K=K,
                xi=xi,
                D=D,
                image_size=image_size,
                yaw_deg=yaw,
                pitch_deg=pitch,
            )

            output = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            output_name = f"test_yaw_{yaw:+04d}_pitch_{pitch:+03d}.jpg"
            output_path = TEST_OUTPUT_DIR / output_name
            cv2.imwrite(str(output_path), output)
            count += 1

    print(f"Saved {count} rotated equirectangular test images.")
    print(f"Output directory: {TEST_OUTPUT_DIR}")


if __name__ == "__main__":
    main()