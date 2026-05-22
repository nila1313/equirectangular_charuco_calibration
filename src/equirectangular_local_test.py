from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import RESULTS_DIR, DETECTION_SUMMARY_PATH
from .calibration import load_useful_frame_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OMNI_CALIBRATION_PATH = RESULTS_DIR / "omni_calibration.npz"
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "equirectangular_local_test"


def load_omni_calibration():
    data = np.load(OMNI_CALIBRATION_PATH, allow_pickle=True)

    K = data["camera_matrix"].astype(np.float64)
    xi = data["xi"].astype(np.float64)
    D = data["dist_coeffs"].astype(np.float64)
    image_size = tuple(int(value) for value in data["image_size"])

    return K, xi, D, image_size


def rotation_matrix_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0):
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


def build_local_lonlat_maps(
    K,
    xi,
    D,
    image_size,
    yaw_deg,
    pitch_deg,
    horizontal_fov_deg,
    vertical_fov_deg,
):
    output_size = (1280, 720)

    R = rotation_matrix_from_euler(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
    )

    horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
    vertical_fov_rad = np.deg2rad(vertical_fov_deg)

    # This is the important part:
    # Smaller FOV = more focused local equirectangular patch.
    P = np.array(
        [
            [output_size[0] / horizontal_fov_rad, 0.0, output_size[0] / 2.0],
            [0.0, output_size[1] / vertical_fov_rad, output_size[1] / 2.0],
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

    return map_x, map_y


def main():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    K, xi, D, image_size = load_omni_calibration()
    frame_paths = load_useful_frame_paths(DETECTION_SUMMARY_PATH)

    if not frame_paths:
        raise RuntimeError("No useful raw frames found.")

    frame_path = frame_paths[0]
    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")

    print(f"Testing frame: {frame_path}")

    yaw_values = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    pitch_values = [-45, -25, 0, 25, 45]

    fov_settings = [
        (140, 90),
        (120, 80),
        (100, 70),
        (80, 60),
    ]

    count = 0

    for horizontal_fov, vertical_fov in fov_settings:
        for yaw in yaw_values:
            for pitch in pitch_values:
                map_x, map_y = build_local_lonlat_maps(
                    K=K,
                    xi=xi,
                    D=D,
                    image_size=image_size,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    horizontal_fov_deg=horizontal_fov,
                    vertical_fov_deg=vertical_fov,
                )

                output = cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                output_name = (
                    f"local_h{horizontal_fov}_v{vertical_fov}"
                    f"_yaw_{yaw:+04d}_pitch_{pitch:+03d}.jpg"
                )
                output_path = TEST_OUTPUT_DIR / output_name
                cv2.imwrite(str(output_path), output)
                count += 1

    print(f"Saved {count} local equirectangular test images.")
    print(f"Output directory: {TEST_OUTPUT_DIR}")


if __name__ == "__main__":
    main()