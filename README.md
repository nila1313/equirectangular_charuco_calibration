# Equirectangular ChArUco Calibration

Pipeline for detecting ChArUco corners in omnidirectional / fisheye video frames,
building a preliminary calibration, rectifying frames, detecting corners again, and
producing a final calibration.

## Pipeline

```bash
python -m src.raw_charuco_detection
python -m src.calibration
python -m src.rectification --alpha 1 --keep-full-frame
python -m src.rectified_charuco_detection
python -m src.final_calibration
python -m src.report
```

## Main Outputs

- `results/detection_summary.csv`
- `results/preliminary_calibration.npz`
- `results/rectification_summary.csv`
- `results/rectified_detection_summary.csv`
- `results/final_calibration.npz`
- `results/calibration_report.pdf`

Large local files such as videos and generated frame images are ignored by Git.
