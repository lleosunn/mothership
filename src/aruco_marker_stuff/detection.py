import cv2
import cv2.aruco as aruco
import numpy as np

def rvec_tvec_to_T(rvec, tvec):
    """Convert OpenCV rvec/tvec to a 4x4 transform matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def invert_T(T):
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def create_detector_parameters():
    """Handle both new and old OpenCV ArUco APIs."""
    if hasattr(aruco, "DetectorParameters"):
        # OpenCV ≥ 4.10
        return aruco.DetectorParameters()
    elif hasattr(aruco, "DetectorParameters_create"):
        # Older OpenCV
        return aruco.DetectorParameters_create()
    else:
        raise RuntimeError("Cannot find DetectorParameters class in cv2.aruco")

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.03):
    """Compatible replacement for aruco.drawAxis."""
    if hasattr(cv2, "drawFrameAxes"):
        # OpenCV ≥ 4.10
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    elif hasattr(aruco, "drawAxis"):
        # Older OpenCV
        aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    else:
        print("⚠️ drawAxis/drawFrameAxes not available in this OpenCV build.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = create_detector_parameters()

    # Example calibration (replace with your actual calibration for accuracy)
    camera_matrix = np.array([[1417.0403096571977, 0.0, 957.459776490542],
                              [0.0, 1420.2720211645985, 570.5911420017325],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.array([
        [0.08959465840442786, -0.750613499067229, 0.00020226506264608155, -0.009160126115283456, 2.7073995410808753]
    ])
    marker_size = 0.200 

    print("📷 Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                marker_id = int(ids[i])
                print(f"\n🟩 Detected Marker ID: {marker_id}")

                marker_points = np.array([
                    [-marker_size/2,  marker_size/2, 0],
                    [ marker_size/2,  marker_size/2, 0],
                    [ marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ], dtype=np.float32)

                image_points = corner.reshape(-1, 2)
                success, rvec, tvec = cv2.solvePnP(marker_points, image_points,
                                                   camera_matrix, dist_coeffs)
                if not success:
                    print("  ❌ Pose estimation failed.")
                    continue

                draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec)

                # Build transform of current marker in camera frame
                T_cam_marker = rvec_tvec_to_T(rvec, tvec)

                # === WORLD FRAME SETUP (Marker 1) ===
                if marker_id == 0:
                    T_world_cam = invert_T(T_cam_marker)
                    have_world = True
                    print("  🌍 Marker 0 sets world frame.")

                # === ROBOT MARKER (Marker 2) ===
                if marker_id == 1:
                    print("  🤖 Marker 1 detected.")
                    if 'have_world' in locals() and have_world:
                        T_world_m2 = T_world_cam @ T_cam_marker
                        pos = T_world_m2[:3, 3]
                        print("  📍 Marker 1 Position in WORLD frame:")
                        print(f"     x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

                        # --- Orientation extraction ---
                        R_m2 = T_world_m2[:3, :3]
                        sy = np.sqrt(R_m2[0,0]**2 + R_m2[1,0]**2)
                        singular = sy < 1e-6

                        if not singular:
                            roll = np.degrees(np.arctan2(R_m2[2,1], R_m2[2,2]))
                            pitch = np.degrees(np.arctan2(-R_m2[2,0], sy))
                            yaw = np.degrees(np.arctan2(R_m2[1,0], R_m2[0,0]))
                        else:
                            roll = np.degrees(np.arctan2(-R_m2[1,2], R_m2[1,1]))
                            pitch = np.degrees(np.arctan2(-R_m2[2,0], sy))
                            yaw = 0

                        print("  🎯 Marker 1 Orientation (WORLD frame, °):")
                        print(f"     roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
                    else:
                        print("  ❌ World frame not established yet — detect Marker 0 first.")

        else:
            cv2.putText(frame, "No markers detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("ArUco Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()