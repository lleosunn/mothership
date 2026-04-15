import cv2
import numpy as np
import os
import json

# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015

# ...
PATH_TO_YOUR_IMAGES = 'charuco_images'
# ------------------------------

def calibrate_and_save_parameters():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
    image_files.sort()

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"  ⚠ Could not read {image_file}, skipping")
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image)

        if charuco_corners is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"  ✓ {image_file}: {len(charuco_corners)} corners")
        else:
            n = 0 if charuco_corners is None else len(charuco_corners)
            print(f"  ✗ {image_file}: only {n} corners, skipping")

    if len(all_charuco_corners) == 0:
        print("\n❌ No charuco corners detected in any image. Check that your board config matches your printed board.")
        return

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None
    )

    # Print calibration results
    print(f"\n{'='*50}")
    print(f"Calibration Results:")
    print(f"{'='*50}")
    print(f"RMS Error: {retval}")
    print(f"Images used: {len(all_charuco_corners)}")
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs}")
    print(f"{'='*50}\n")

    # Save calibration data to JSON
    calibration_data = {
        "rms_error": retval,
        "images_used": len(all_charuco_corners),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "board_config": {
            "squares_vertically": SQUARES_VERTICALLY,
            "squares_horizontally": SQUARES_HORIZONTALLY,
            "square_length_meters": SQUARE_LENGTH,
            "marker_length_meters": MARKER_LENGTH,
            "aruco_dict": "DICT_6X6_250"
        }
    }
    
    with open('calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print("✅ Calibration data saved to calibration.json")

    # Iterate through displaying all the images
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

calibrate_and_save_parameters()