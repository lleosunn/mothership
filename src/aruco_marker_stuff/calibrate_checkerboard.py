#!/usr/bin/env python3
import cv2
import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Calibrate camera from checkerboard images")
parser.add_argument("--rows", type=int, default=8, help="Inner corner rows (squares - 1)")
parser.add_argument("--cols", type=int, default=5, help="Inner corner cols (squares - 1)")
parser.add_argument("--square-size", type=float, default=0.03, help="Square size in meters")
parser.add_argument("--images", type=str, default="charuco_images", help="Directory with captured .png images")
parser.add_argument("-o", "--output", type=str, default="calibration.json", help="Output calibration file")
args = parser.parse_args()

PATTERN_SIZE = (args.cols, args.rows)
SQUARE_SIZE = args.square_size
PATH_TO_IMAGES = args.images

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

obj_points = []
img_points = []
image_size = None

image_files = sorted([
    os.path.join(PATH_TO_IMAGES, f)
    for f in os.listdir(PATH_TO_IMAGES)
    if f.lower().endswith(".png")
])

if not image_files:
    print(f"❌ No .png images found in {PATH_TO_IMAGES}/")
    exit(1)

print(f"Processing {len(image_files)} images looking for {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} inner corners...\n")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"  ⚠ Could not read {image_file}, skipping")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)
        image_size = gray.shape[::-1]
        print(f"  ✓ {image_file}")
    else:
        print(f"  ✗ {image_file}: pattern not found")

print(f"\n{len(img_points)}/{len(image_files)} images usable")

if len(img_points) == 0:
    print("❌ No checkerboard detected in any image. Verify --rows/--cols match your board.")
    exit(1)

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)

print(f"\n{'='*50}")
print(f"Calibration Results:")
print(f"{'='*50}")
print(f"RMS Error: {retval:.4f}")
print(f"Images used: {len(img_points)}")
print(f"\nCamera Matrix:\n{camera_matrix}")
print(f"\nDistortion Coefficients:\n{dist_coeffs}")
print(f"{'='*50}\n")

calibration_data = {
    "rms_error": retval,
    "images_used": len(img_points),
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": dist_coeffs.tolist(),
    "board_config": {
        "inner_corners_cols": PATTERN_SIZE[0],
        "inner_corners_rows": PATTERN_SIZE[1],
        "square_size_meters": SQUARE_SIZE,
    }
}

with open(args.output, 'w') as f:
    json.dump(calibration_data, f, indent=4)

print(f"✅ Calibration data saved to {args.output}")

print("\nShowing undistorted images (press any key to advance, 'q' to quit)...")
for image_file in image_files:
    image = cv2.imread(image_file)
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    cv2.imshow('Undistorted', undistorted)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Done.")
