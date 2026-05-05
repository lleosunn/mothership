#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from functools import partial
import time


def rvec_tvec_to_T(rvec, tvec):
    """Convert OpenCV rvec/tvec to a 4x4 transform matrix."""
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.flatten()
    return T 
    
def invert_T(T):
    """Invert a 4x4 rigid transform."""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3, :3] = R_mat.T
    Tinv[:3, 3] = -R_mat.T @ t
    return Tinv


def slerp_quaternion(q1, q2, t):
    """Spherical linear interpolation between two quaternions (xyzw format)."""
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Compute slerp
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q_perp = q2 - q1 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)
    
    return q1 * np.cos(theta) + q_perp * np.sin(theta)


class MultiCameraDetectionNode(Node):
    def __init__(self):
        super().__init__('multi_camera_detection')

        self.declare_parameter('num_cameras', 2)
        self.num_cameras = self.get_parameter('num_cameras').value

        self.agent_ids = [1, 2, 3, 4, 7]
        self.bridge = CvBridge()

        # Timeout in seconds - if no message received within this time, consider camera as not seeing marker
        self.pose_timeout = 0.5

        # Storage for poses from each camera for each agent
        # camera_poses[camera_id][agent_id] = {'pose': Pose, 'timestamp': float}
        self.camera_poses = {
            cam: {i: {'pose': None, 'timestamp': 0.0} for i in self.agent_ids}
            for cam in range(self.num_cameras)
        }

        # Publishers for fused poses (one per agent), keyed by marker ID
        self.fused_pose_pubs = {
            i: self.create_publisher(PoseStamped, f'/fused_pose_{i}', 10)
            for i in self.agent_ids
        }

        # Per-camera world frame transforms
        self.world_transforms = {
            cam: {'have_world': False, 'T_world_cam': None}
            for cam in range(self.num_cameras)
        }

        # Subscribers for camera image topics (created in a loop)
        self.camera_subs = {}
        for cam in range(self.num_cameras):
            self.camera_subs[cam] = self.create_subscription(
                Image,
                f'/camera{cam}/image_raw',
                partial(self._camera_callback, camera_id=cam),
                10
            )

        # --- ArUco setup ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = self._create_detector_parameters()

        # Camera calibration (same for all cameras - adjust if different)
        self.camera_matrix = np.array([
            [192.60922574, 0.0, 325.85340988],
            [0.0, 191.78659388, 276.03173316],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            [
                -0.04275189,
                0.00461634,
                -0.01543687,
                0.00461978,
                0.00053686
            ]
        ], dtype=np.float32)

        self.marker_size = 0.1585  # units are in meters

        # Timer as fallback to ensure output even when detections are sparse
        self.fusion_timer = self.create_timer(0.1, self.publish_fused_poses)

        topics = ", ".join(f"/camera{c}/image_raw" for c in range(self.num_cameras))
        self.get_logger().info(f"📷 Multi Camera Detection Node started ({self.num_cameras} cameras).")
        self.get_logger().info(f"   Subscribing to {topics}")
        self.get_logger().info("   Publishing fused poses to /fused_pose_*")

    def _create_detector_parameters(self):
        if hasattr(aruco, "DetectorParameters"):
            return aruco.DetectorParameters()
        elif hasattr(aruco, "DetectorParameters_create"):
            return aruco.DetectorParameters_create()
        else:
            raise RuntimeError("Cannot find DetectorParameters in cv2.aruco")

    def _camera_callback(self, msg, camera_id):
        """Process images from any camera."""
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"❌ Failed to convert camera{camera_id} image: {e}")
            return

        wt = self.world_transforms[camera_id]
        self._process_frame(
            gray,
            camera_id=camera_id,
            have_world=wt['have_world'],
            T_world_cam=wt['T_world_cam'],
        )

    def _process_frame(self, gray, camera_id, have_world, T_world_cam):
        """Process a grayscale frame for ArUco detection and store poses."""
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is None:
            return

        # Marker geometry for solvePnP
        marker_points = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ], dtype=np.float32)

        for i, corner in enumerate(corners):
            marker_id = int(ids[i])
            image_points = corner.reshape(-1, 2)

            success, rvec, tvec = cv2.solvePnP(
                marker_points, image_points,
                self.camera_matrix, self.dist_coeffs
            )
            if not success:
                continue

            T_cam_marker = rvec_tvec_to_T(rvec, tvec)

            # Marker 0 = WORLD FRAME ORIGIN
            if marker_id == 0:
                new_T_world_cam = invert_T(T_cam_marker)
                self.world_transforms[camera_id]['have_world'] = True
                self.world_transforms[camera_id]['T_world_cam'] = new_T_world_cam
                T_world_cam = new_T_world_cam
                have_world = True
                self.get_logger().info(f"🌍 Camera{camera_id}: Marker 0 sets WORLD frame.")

            # ROBOT MARKERS
            if have_world and T_world_cam is not None and marker_id in self.agent_ids:
                T_world_m = T_world_cam @ T_cam_marker
                pos = T_world_m[:3, 3]
                R_mat = T_world_m[:3, :3]
                rot = R.from_matrix(R_mat)
                q_xyzw = rot.as_quat()
                roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

                pose_msg = Pose()
                pose_msg.position.x = float(pos[0])
                pose_msg.position.y = float(pos[1])
                pose_msg.position.z = float(pos[2])
                pose_msg.orientation.x = float(q_xyzw[0])
                pose_msg.orientation.y = float(q_xyzw[1])
                pose_msg.orientation.z = float(q_xyzw[2])
                pose_msg.orientation.w = float(q_xyzw[3])

                cam_to_marker_dist = float(np.linalg.norm(tvec))

                self.camera_poses[camera_id][marker_id] = {
                    'pose': pose_msg,
                    'timestamp': time.time(),
                    'distance': cam_to_marker_dist,
                }

                self.get_logger().debug(
                    f"🤖 Camera{camera_id} Marker {marker_id} WORLD pose: "
                    f"x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}, "
                    f"yaw={yaw:.1f}°"
                )

        # Publish immediately after processing frame (reduces latency)
        self.publish_fused_poses()

    def publish_fused_poses(self):
        """Compute and publish fused poses for each agent using all valid cameras."""
        current_time = time.time()

        for agent_id in self.agent_ids:
            valid_data = []
            for cam_id in range(self.num_cameras):
                data = self.camera_poses[cam_id][agent_id]
                if (data['pose'] is not None and
                        (current_time - data['timestamp']) < self.pose_timeout):
                    valid_data.append((cam_id, data))

            if not valid_data:
                continue

            if len(valid_data) == 1:
                fused_pose = valid_data[0][1]['pose']
            else:
                # Inverse-distance weights: closer camera gets more influence
                weights = []
                for _, data in valid_data:
                    d = data.get('distance', 1.0)
                    weights.append(1.0 / max(d, 1e-6))
                total_w = sum(weights)
                weights = [w / total_w for w in weights]

                fused_pose = Pose()
                fused_pose.position.x = sum(w * d['pose'].position.x for w, (_, d) in zip(weights, valid_data))
                fused_pose.position.y = sum(w * d['pose'].position.y for w, (_, d) in zip(weights, valid_data))
                fused_pose.position.z = sum(w * d['pose'].position.z for w, (_, d) in zip(weights, valid_data))

                # Iterative pairwise slerp for quaternion fusion
                q_fused = [
                    valid_data[0][1]['pose'].orientation.x,
                    valid_data[0][1]['pose'].orientation.y,
                    valid_data[0][1]['pose'].orientation.z,
                    valid_data[0][1]['pose'].orientation.w,
                ]
                cumulative_w = weights[0]
                for k in range(1, len(valid_data)):
                    q_next = [
                        valid_data[k][1]['pose'].orientation.x,
                        valid_data[k][1]['pose'].orientation.y,
                        valid_data[k][1]['pose'].orientation.z,
                        valid_data[k][1]['pose'].orientation.w,
                    ]
                    cumulative_w += weights[k]
                    t = weights[k] / cumulative_w
                    q_fused = slerp_quaternion(q_fused, q_next, t)

                fused_pose.orientation.x = float(q_fused[0])
                fused_pose.orientation.y = float(q_fused[1])
                fused_pose.orientation.z = float(q_fused[2])
                fused_pose.orientation.w = float(q_fused[3])

            stamped = PoseStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = 'world'
            stamped.pose = fused_pose
            self.fused_pose_pubs[agent_id].publish(stamped)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
