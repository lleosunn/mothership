#!/usr/bin/env python3
"""Save webcam frames for checkerboard calibration (same inner-corner counts as calibrate_checkerboard.py)."""
import argparse
import glob
import os

import cv2

parser = argparse.ArgumentParser(description="Capture checkerboard images from a camera for calibration")
parser.add_argument("--output-dir", type=str, default="my_cal_images", help="Where to save .png frames")
parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (often 0)")
parser.add_argument("--rows", type=int, default=8, help="Inner corner rows (must match calibrate_checkerboard.py)")
parser.add_argument("--cols", type=int, default=5, help="Inner corner cols (must match calibrate_checkerboard.py)")
args = parser.parse_args()

pattern_size = (args.cols, args.rows)
os.makedirs(args.output_dir, exist_ok=True)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Could not open camera index {args.camera}. Try --camera 1 or another index.")
    exit(1)

print(f"Saving to {args.output_dir}/  (pattern inner corners: {pattern_size[0]}×{pattern_size[1]})")
print("Hold your printed checkerboard in view. Green overlay = pattern detected.")
print("Press 's' to save a frame, 'q' to quit. Aim for 20+ varied poses (angles, distance, edges of frame).")

existing = glob.glob(os.path.join(args.output_dir, "checker_[0-9][0-9][0-9].png"))
count = 0
for path in existing:
    base = os.path.basename(path)
    try:
        count = max(count, int(base.split("_")[1].split(".")[0]) + 1)
    except (IndexError, ValueError):
        pass

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed.")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, pattern_size, corners, found)

    cv2.imshow("Checkerboard capture (s=save, q=quit)", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        path = os.path.join(args.output_dir, f"checker_{count:03d}.png")
        cv2.imwrite(path, frame)
        status = "✓ pattern visible" if found else "⚠ pattern not detected in this frame"
        print(f"Saved {path} ({status})")
        count += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
