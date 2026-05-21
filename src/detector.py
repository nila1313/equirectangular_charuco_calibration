from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DETECTED_FRAMES_DIR,
    DETECTION_SUMMARY_PATH,
    RAW_FRAMES_DIR,
    BoardConfig,
    RawDetectionConfig,
    ensure_project_dirs,
)


@dataclass(frozen=True)
class DetectionResult:
    frame_index: int
    raw_frame_path: Path
    detected_frame_path: Path | None
    marker_count: int
    charuco_corner_count: int
    marker_ids: str
    used_for_calibration: bool
    error: str


@dataclass(frozen=True)
class CharucoDetection:
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray | None
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    error: str = ""

    @property
    def marker_count(self) -> int:
        return 0 if self.marker_ids is None else len(self.marker_ids)

    @property
    def charuco_corner_count(self) -> int:
        return 0 if self.charuco_ids is None else len(self.charuco_ids)

    @property
    def marker_ids_text(self) -> str:
        if self.marker_ids is None:
            return ""
        return " ".join(str(int(marker_id)) for marker_id in self.marker_ids.flatten())


def get_aruco_dictionary(dictionary_name: str) -> cv2.aruco.Dictionary:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        valid_names = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise ValueError(
            f"Unknown ArUco dictionary {dictionary_name!r}. "
            f"Examples: {', '.join(valid_names[:8])}"
        )
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_charuco_board(board_config: BoardConfig) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (board_config.squares_x, board_config.squares_y),
        board_config.square_length,
        board_config.marker_length,
        get_aruco_dictionary(board_config.dictionary_name),
    )


class CharucoDetector:
    def __init__(self, board_config: BoardConfig) -> None:
        self.board_config = board_config
        self.dictionary = get_aruco_dictionary(board_config.dictionary_name)
        self.board = create_charuco_board(board_config)
        self.allowed_marker_ids = set(int(marker_id) for marker_id in self.board.getIds().flatten())
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            detector_params,
        )

    def detect(self, frame: np.ndarray) -> CharucoDetection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected_corners = self.detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            return CharucoDetection(marker_corners, marker_ids, None, None)

        marker_corners, marker_ids = self._refine_and_filter_markers(
            gray,
            marker_corners,
            marker_ids,
            rejected_corners,
        )
        if marker_ids is None or len(marker_ids) == 0:
            return CharucoDetection(marker_corners, marker_ids, None, None)

        try:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
            )
        except cv2.error as exc:
            return CharucoDetection(
                marker_corners,
                marker_ids,
                None,
                None,
                str(exc).splitlines()[0],
            )

        return CharucoDetection(marker_corners, marker_ids, charuco_corners, charuco_ids)

    def _refine_and_filter_markers(
        self,
        gray: np.ndarray,
        marker_corners: tuple[np.ndarray, ...],
        marker_ids: np.ndarray,
        rejected_corners: tuple[np.ndarray, ...],
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
        try:
            marker_corners, marker_ids, _, _ = self.detector.refineDetectedMarkers(
                gray,
                self.board,
                marker_corners,
                marker_ids,
                rejected_corners,
            )
        except cv2.error:
            pass

        if marker_ids is None or len(marker_ids) == 0:
            return tuple(), None

        filtered = [
            (corner, marker_id)
            for corner, marker_id in zip(marker_corners, marker_ids)
            if int(marker_id[0]) in self.allowed_marker_ids
        ]
        if not filtered:
            return tuple(), None

        filtered_corners, filtered_ids = zip(*filtered)
        return tuple(filtered_corners), np.array(filtered_ids, dtype=marker_ids.dtype)

    def draw(self, frame: np.ndarray, detection: CharucoDetection) -> np.ndarray:
        annotated = frame.copy()
        if detection.marker_ids is not None and detection.marker_count > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                detection.marker_corners,
                detection.marker_ids,
            )
        if detection.charuco_ids is not None and detection.charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                detection.charuco_corners,
                detection.charuco_ids,
            )
        return annotated


def detect_and_save_frame(
    frame: np.ndarray,
    frame_index: int,
    detector: CharucoDetector,
    config: RawDetectionConfig,
) -> DetectionResult:
    raw_frame_path = RAW_FRAMES_DIR / f"frame_{frame_index:06d}.jpg"
    detected_frame_path = DETECTED_FRAMES_DIR / f"frame_{frame_index:06d}_detected.jpg"
    cv2.imwrite(str(raw_frame_path), frame)

    detection = detector.detect(frame)
    if detection.marker_count > 0:
        annotated = detector.draw(frame, detection)
        cv2.imwrite(str(detected_frame_path), annotated)
    else:
        detected_frame_path = None

    return DetectionResult(
        frame_index=frame_index,
        raw_frame_path=raw_frame_path,
        detected_frame_path=detected_frame_path,
        marker_count=detection.marker_count,
        charuco_corner_count=detection.charuco_corner_count,
        marker_ids=detection.marker_ids_text,
        used_for_calibration=(
            detection.marker_count >= config.min_markers
            and detection.charuco_corner_count > 0
        ),
        error=detection.error,
    )


def run_raw_detection(config: RawDetectionConfig) -> list[DetectionResult]:
    ensure_project_dirs()

    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")

    detector = CharucoDetector(config.board)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    results: list[DetectionResult] = []
    processed_count = 0

    for frame_index in range(0, total_frames, config.frame_step):
        if config.max_frames is not None and processed_count >= config.max_frames:
            break

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            results.append(
                DetectionResult(
                    frame_index=frame_index,
                    raw_frame_path=RAW_FRAMES_DIR / f"frame_{frame_index:06d}.jpg",
                    detected_frame_path=None,
                    marker_count=0,
                    charuco_corner_count=0,
                    marker_ids="",
                    used_for_calibration=False,
                    error="Could not read frame",
                )
            )
            continue

        results.append(detect_and_save_frame(frame, frame_index, detector, config))
        processed_count += 1

    capture.release()
    write_detection_summary(results)
    return results


def write_detection_summary(results: list[DetectionResult]) -> None:
    with DETECTION_SUMMARY_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "raw_frame_path",
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
                    "frame_index": result.frame_index,
                    "raw_frame_path": result.raw_frame_path,
                    "detected_frame_path": result.detected_frame_path or "",
                    "marker_count": result.marker_count,
                    "charuco_corner_count": result.charuco_corner_count,
                    "marker_ids": result.marker_ids,
                    "used_for_calibration": result.used_for_calibration,
                    "error": result.error,
                }
            )
