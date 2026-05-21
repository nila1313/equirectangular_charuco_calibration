from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "video"
RAW_FRAMES_DIR = DATA_DIR / "raw_frames"
DETECTED_FRAMES_DIR = DATA_DIR / "detected_frames"
RECTIFIED_FRAMES_DIR = DATA_DIR / "rectified_frames"
RECTIFIED_DETECTED_FRAMES_DIR = DATA_DIR / "rectified_detected_frames"
RESULTS_DIR = PROJECT_ROOT / "results"

DETECTION_SUMMARY_PATH = RESULTS_DIR / "detection_summary.csv"
RECTIFICATION_SUMMARY_PATH = RESULTS_DIR / "rectification_summary.csv"
RECTIFIED_DETECTION_SUMMARY_PATH = RESULTS_DIR / "rectified_detection_summary.csv"
PRELIMINARY_CALIBRATION_PATH = RESULTS_DIR / "preliminary_calibration.npz"
FINAL_CALIBRATION_PATH = RESULTS_DIR / "final_calibration.npz"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class BoardConfig:
    squares_x: int = 7
    squares_y: int = 7
    square_length: float = 0.14285714285714285
    marker_length: float = 0.10
    dictionary_name: str = "DICT_4X4_1000"


@dataclass(frozen=True)
class RawDetectionConfig:
    video_path: Path
    board: BoardConfig = BoardConfig()
    frame_step: int = 30
    max_frames: int | None = None
    min_markers: int = 6


@dataclass(frozen=True)
class CalibrationConfig:
    board: BoardConfig = BoardConfig()
    min_corners: int = 9
    output_path: Path = PRELIMINARY_CALIBRATION_PATH


def ensure_project_dirs() -> None:
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    DETECTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    RECTIFIED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    RECTIFIED_DETECTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_default_video() -> Path:
    videos = sorted(
        path for path in VIDEO_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video file found in {VIDEO_DIR}")
    if len(videos) > 1:
        print(f"Multiple videos found; using {videos[0].name}")
    return videos[0]
