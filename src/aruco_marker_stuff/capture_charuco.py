#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import os


# === SETUP ===
output_dir = "charuco_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("📷 Press 's' to save a frame, 'q' to quit.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("ChArUco Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(output_dir, f"charuco_{count:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("📦 Done capturing images!")