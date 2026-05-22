import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH SETUP
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

PERSPECTIVE_EXPERIMENT_DIR = os.path.join(
    RESULTS_DIR,
    "perspective_detection_experiment"
)

INPUT_SUMMARY_CSV = os.path.join(
    PERSPECTIVE_EXPERIMENT_DIR,
    "perspective_detection_summary.csv"
)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "hybrid_detection_selection")

SELECTED_DETECTION_DIR = os.path.join(OUTPUT_DIR, "selected_detection")
SELECTED_RAW_DIR = os.path.join(OUTPUT_DIR, "selected_raw")
SELECTED_PERSPECTIVE_DIR = os.path.join(OUTPUT_DIR, "selected_perspective")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SELECTED_DETECTION_DIR, exist_ok=True)
os.makedirs(SELECTED_RAW_DIR, exist_ok=True)
os.makedirs(SELECTED_PERSPECTIVE_DIR, exist_ok=True)


# ============================================================
# 2. SELECTION RULES
# ============================================================

# Minimum number of ChArUco corners to consider a frame useful.
# You can adjust this later.
MIN_CHARUCO_CORNERS = 4

# If raw and perspective have the same number of ChArUco corners,
# prefer raw because it avoids extra rectification artifacts.
PREFER_RAW_ON_TIE = True


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_copy(src_path, dst_path):
    if isinstance(src_path, str) and os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        return True
    return False


def choose_best_detection(row):
    raw_corners = int(row["raw_charuco_corners"])
    persp_corners = int(row["perspective_charuco_corners"])

    if persp_corners > raw_corners:
        selected_source = "perspective"
        selected_corners = persp_corners
        selected_markers = int(row["perspective_markers"])
        selected_detection_image = row["perspective_detection_image"]
        selected_rectified_image = row["perspective_image"]
    elif raw_corners > persp_corners:
        selected_source = "raw"
        selected_corners = raw_corners
        selected_markers = int(row["raw_markers"])
        selected_detection_image = row["raw_detection_image"]
        selected_rectified_image = ""
    else:
        if PREFER_RAW_ON_TIE:
            selected_source = "raw"
            selected_corners = raw_corners
            selected_markers = int(row["raw_markers"])
            selected_detection_image = row["raw_detection_image"]
            selected_rectified_image = ""
        else:
            selected_source = "perspective"
            selected_corners = persp_corners
            selected_markers = int(row["perspective_markers"])
            selected_detection_image = row["perspective_detection_image"]
            selected_rectified_image = row["perspective_image"]

    useful = selected_corners >= MIN_CHARUCO_CORNERS

    return {
        "selected_source": selected_source,
        "selected_markers": selected_markers,
        "selected_charuco_corners": selected_corners,
        "selected_detection_image": selected_detection_image,
        "selected_rectified_image": selected_rectified_image,
        "useful_for_calibration": useful,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("Starting hybrid detection selection...")
    print("Input summary:", INPUT_SUMMARY_CSV)
    print("Output directory:", OUTPUT_DIR)

    if not os.path.exists(INPUT_SUMMARY_CSV):
        raise FileNotFoundError(
            f"Could not find input CSV:\n{INPUT_SUMMARY_CSV}\n\n"
            "Run src/perspective_detection_experiment.py first."
        )

    df = pd.read_csv(INPUT_SUMMARY_CSV)

    selected_rows = []

    for _, row in df.iterrows():
        frame_name = row["frame"]

        selected = choose_best_detection(row)

        output_detection_name = (
            f"{frame_name}_selected_{selected['selected_source']}_detection.jpg"
        )

        output_detection_path = os.path.join(
            SELECTED_DETECTION_DIR,
            output_detection_name
        )

        copied_detection = safe_copy(
            selected["selected_detection_image"],
            output_detection_path
        )

        selected_rectified_output_path = ""

        if selected["selected_source"] == "raw":
            raw_output_path = os.path.join(
                SELECTED_RAW_DIR,
                f"{frame_name}_raw_selected_detection.jpg"
            )
            safe_copy(selected["selected_detection_image"], raw_output_path)

        elif selected["selected_source"] == "perspective":
            persp_output_path = os.path.join(
                SELECTED_PERSPECTIVE_DIR,
                f"{frame_name}_perspective_selected_detection.jpg"
            )
            safe_copy(selected["selected_detection_image"], persp_output_path)

            if isinstance(selected["selected_rectified_image"], str) and os.path.exists(
                selected["selected_rectified_image"]
            ):
                selected_rectified_output_path = os.path.join(
                    SELECTED_PERSPECTIVE_DIR,
                    f"{frame_name}_perspective_selected_rectified.jpg"
                )
                safe_copy(
                    selected["selected_rectified_image"],
                    selected_rectified_output_path
                )

        selected_rows.append({
            "frame": frame_name,
            "source_path": row["source_path"],

            "raw_markers": int(row["raw_markers"]),
            "raw_charuco_corners": int(row["raw_charuco_corners"]),

            "perspective_markers": int(row["perspective_markers"]),
            "perspective_charuco_corners": int(row["perspective_charuco_corners"]),

            "selected_source": selected["selected_source"],
            "selected_markers": selected["selected_markers"],
            "selected_charuco_corners": selected["selected_charuco_corners"],

            "gain_over_raw": (
                selected["selected_charuco_corners"]
                - int(row["raw_charuco_corners"])
            ),

            "useful_for_calibration": selected["useful_for_calibration"],

            "selected_detection_image": output_detection_path
            if copied_detection else selected["selected_detection_image"],

            "selected_rectified_image": selected_rectified_output_path,
        })

    summary = pd.DataFrame(selected_rows)

    summary_path = os.path.join(OUTPUT_DIR, "hybrid_detection_summary.csv")
    useful_path = os.path.join(OUTPUT_DIR, "hybrid_useful_frames.csv")

    summary.to_csv(summary_path, index=False)

    useful_df = summary[summary["useful_for_calibration"] == True].copy()
    useful_df.to_csv(useful_path, index=False)

    print("\n======================================")
    print("DONE")
    print("======================================")
    print("Hybrid summary saved to:", summary_path)
    print("Useful frames saved to:", useful_path)

    print("\nSelection source counts:")
    print(summary["selected_source"].value_counts())

    print("\nUseful frame count:")
    print(len(useful_df), "out of", len(summary))

    print("\nTotal ChArUco corners:")
    print("Raw total:", summary["raw_charuco_corners"].sum())
    print("Perspective total:", summary["perspective_charuco_corners"].sum())
    print("Hybrid selected total:", summary["selected_charuco_corners"].sum())

    print("\nTop hybrid gains:")
    print(
        summary.sort_values("gain_over_raw", ascending=False)
        [[
            "frame",
            "raw_charuco_corners",
            "perspective_charuco_corners",
            "selected_source",
            "selected_charuco_corners",
            "gain_over_raw",
            "useful_for_calibration",
        ]]
        .head(20)
    )


if __name__ == "__main__":
    main()