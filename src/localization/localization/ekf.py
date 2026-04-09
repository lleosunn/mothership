#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q: Quaternion) -> float:
    """
    Extract yaw (rotation around Z) from a quaternion.
    """
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quaternion:
    """
    Create a quaternion representing a yaw rotation (no roll/pitch).
    """
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class RobotEkfState:
    """Holds EKF state for a single robot."""
    def __init__(self):
        # EKF state: [x, y, yaw]^T
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 1.0
        self.initialized = False
        
        # Latest twist (body frame)
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        
        self.last_predict_time = None
        self.last_z_height = 0.0


class SimpleEkfFuser(Node):
    """
    EKF to fuse poses and twists for multiple robots:
      - /pose_N  (geometry_msgs/Pose)   -> noisy global pose
      - /twist_N (geometry_msgs/Twist)  -> accurate body-frame velocities

    Publishes:
      - /fused_pose_N (geometry_msgs/PoseStamped) -> smoothed global pose
    """

    def __init__(self):
        super().__init__("simple_ekf_fuser")

        # Number of robots to support
        self.num_robots = 2

        # increase Q: trust cameras more (faster response, more noise)
        self.q_x = 0.01   # m^2 per step
        self.q_y = 0.01
        self.q_yaw = 0.01  # rad^2 per step

        # increase R: trust model more (smoother, slower response)
        self.r_x = 0.01   # m^2
        self.r_y = 0.01
        self.r_yaw = 0.01  # rad^2

        # Per-robot EKF state
        self.robot_states = {i: RobotEkfState() for i in range(1, self.num_robots + 1)}

        # Subscribers & publishers for each robot
        self.pose_subs = []
        self.twist_subs = []
        self.pose_pubs = {}

        for robot_id in range(1, self.num_robots + 1):
            # Pose subscriber
            pose_sub = self.create_subscription(
                Pose,
                f'/pose_{robot_id}',
                lambda msg, rid=robot_id: self.pose_callback(msg, rid),
                10
            )
            self.pose_subs.append(pose_sub)

            # Twist subscriber
            twist_sub = self.create_subscription(
                Twist,
                f'/twist_{robot_id}',
                lambda msg, rid=robot_id: self.twist_callback(msg, rid),
                50
            )
            self.twist_subs.append(twist_sub)

            pose_pub = self.create_publisher(PoseStamped, f'/fused_pose_{robot_id}', 10)
            self.pose_pubs[robot_id] = pose_pub

        # Prediction timer (50 Hz)
        self.timer = self.create_timer(0.02, self.predict_timer_callback)

        self.get_logger().info("Simple EKF fuser started for multiple robots:")
        for robot_id in range(1, self.num_robots + 1):
            self.get_logger().info(
                f"  Robot {robot_id}: /pose_{robot_id} + /twist_{robot_id} -> /fused_pose_{robot_id}"
            )

    # ---------- Callbacks ----------

    def twist_callback(self, msg: Twist, robot_id: int):
        """Store latest body-frame velocities for a robot."""
        state = self.robot_states[robot_id]
        state.vx = float(msg.linear.x)
        state.vy = float(msg.linear.y)
        state.wz = float(msg.angular.z)

    def pose_callback(self, msg: Pose, robot_id: int):
        """Process pose measurement for a robot."""
        state = self.robot_states[robot_id]
        yaw_meas = yaw_from_quat(msg.orientation)

        # Initialize filter on first pose
        if not state.initialized:
            state.x[0, 0] = float(msg.position.x)
            state.x[1, 0] = float(msg.position.y)
            state.x[2, 0] = float(yaw_meas)
            state.P = np.eye(3) * 0.5
            state.initialized = True
            state.last_z_height = msg.position.z
            self.get_logger().info(f"EKF initialized for robot {robot_id} from first /pose_{robot_id} measurement.")
            self.publish_fused_pose(robot_id)
            return

        # Measurement vector z = [x, y, yaw]^T
        z = np.array([
            [float(msg.position.x)],
            [float(msg.position.y)],
            [float(yaw_meas)],
        ])

        # Measurement model h(x) = x (we directly observe x,y,yaw)
        H = np.eye(3)

        # Measurement noise covariance R
        R = np.diag([self.r_x, self.r_y, self.r_yaw])

        # Innovation y = z - x
        y = z - state.x
        y[2, 0] = wrap_angle(y[2, 0])  # wrap yaw innovation

        # Innovation covariance S
        S = H @ state.P @ H.T + R

        # Kalman gain
        K = state.P @ H.T @ np.linalg.inv(S)

        # State update
        state.x = state.x + K @ y
        state.x[2, 0] = wrap_angle(state.x[2, 0])

        # Covariance update
        I = np.eye(3)
        state.P = (I - K @ H) @ state.P

        # Store height for publishing
        state.last_z_height = msg.position.z
        self.publish_fused_pose(robot_id)

    def predict_timer_callback(self):
        """Run prediction step for all initialized robots."""
        now = self.get_clock().now().nanoseconds * 1e-9

        for robot_id in range(1, self.num_robots + 1):
            state = self.robot_states[robot_id]
            
            if not state.initialized:
                continue

            if state.last_predict_time is None:
                state.last_predict_time = now
                continue

            dt = now - state.last_predict_time
            if dt <= 0.0 or dt > 0.5:
                # If dt is crazy (e.g. big jump), reset and skip this step
                state.last_predict_time = now
                continue

            state.last_predict_time = now
            self.predict(robot_id, dt)
            # Publishing here keeps the pose moving smoothly even between global updates
            self.publish_fused_pose(robot_id)

    # ---------- EKF Core ----------

    def predict(self, robot_id: int, dt: float):
        """
        EKF prediction step using body-frame velocities.
        """
        state = self.robot_states[robot_id]
        yaw = float(state.x[2, 0])

        # Body-frame v -> global-frame displacement
        dx = (state.vx * math.cos(yaw) - state.vy * math.sin(yaw)) * dt
        dy = (state.vx * math.sin(yaw) + state.vy * math.cos(yaw)) * dt
        dyaw = state.wz * dt

        # State prediction
        state.x[0, 0] += dx
        state.x[1, 0] += dy
        state.x[2, 0] = wrap_angle(state.x[2, 0] + dyaw)

        # Jacobian F = ∂f/∂x
        F = np.eye(3)
        F[0, 2] = (-state.vx * math.sin(yaw) - state.vy * math.cos(yaw)) * dt
        F[1, 2] = ( state.vx * math.cos(yaw) - state.vy * math.sin(yaw)) * dt

        # Process noise
        Q = np.diag([self.q_x, self.q_y, self.q_yaw])

        # Covariance prediction
        state.P = F @ state.P @ F.T + Q

    # ---------- Output ----------

    def publish_fused_pose(self, robot_id: int):
        """
        Publish fused pose for a robot.
        """
        state = self.robot_states[robot_id]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(state.x[0, 0])
        msg.pose.position.y = float(state.x[1, 0])
        msg.pose.position.z = float(state.last_z_height)
        msg.pose.orientation = quat_from_yaw(float(state.x[2, 0]))

        self.pose_pubs[robot_id].publish(msg)


def main():
    rclpy.init()
    node = SimpleEkfFuser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
