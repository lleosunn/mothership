#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TwistStamped, QuaternionStamped
import math
import time


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (in radians)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class RobotVelocityData:
    """Holds velocity fusion data for a single robot."""
    def __init__(self):
        self.linear_vel = None
        self.prev_rpy = None
        self.prev_rpy_time = None
        self.angular_vel = (0.0, 0.0, 0.0)


class VelocityFusionNode(Node):
    def __init__(self):
        super().__init__('velocity_fusion')

        # Number of robots to support
        self.num_robots = 2

        # QoS profile to match the robot (best effort reliability)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Per-robot data storage
        self.robot_data = {i: RobotVelocityData() for i in range(1, self.num_robots + 1)}

        # Create subscribers and publishers for each robot
        self.vel_subs = []
        self.attitude_subs = []
        self.twist_pubs = {}

        for robot_id in range(1, self.num_robots + 1):
            # Subscriber for velocity
            vel_sub = self.create_subscription(
                TwistStamped,
                f'/robomaster_{robot_id}/vel',
                lambda msg, rid=robot_id: self.vel_callback(msg, rid),
                qos_profile
            )
            self.vel_subs.append(vel_sub)

            # Subscriber for attitude
            attitude_sub = self.create_subscription(
                QuaternionStamped,
                f'/robomaster_{robot_id}/attitude',
                lambda msg, rid=robot_id: self.attitude_callback(msg, rid),
                qos_profile
            )
            self.attitude_subs.append(attitude_sub)

            # Publisher for fused twist
            twist_pub = self.create_publisher(
                Twist,
                f'/twist_{robot_id}',
                10
            )
            self.twist_pubs[robot_id] = twist_pub

        # Timer to publish fused twist at regular interval
        self.publish_timer = self.create_timer(0.02, self.publish_fused_twists)  # 50 Hz

        self.get_logger().info("🔄 Velocity Fusion Node started.")
        for robot_id in range(1, self.num_robots + 1):
            self.get_logger().info(
                f"   Robot {robot_id}: /robomaster_{robot_id}/vel + /robomaster_{robot_id}/attitude -> /twist_{robot_id}"
            )

    def vel_callback(self, msg, robot_id):
        """Store linear velocity from /vel topic."""
        self.robot_data[robot_id].linear_vel = (
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        )

    def attitude_callback(self, msg, robot_id):
        """Compute angular velocity from attitude derivative."""
        data = self.robot_data[robot_id]
        q = msg.quaternion
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        current_time = time.time()

        if data.prev_rpy is not None and data.prev_rpy_time is not None:
            dt = current_time - data.prev_rpy_time
            if dt > 0.001:  # Avoid division by very small dt
                # Compute angular velocity as derivative of RPY
                roll_rate = (roll - data.prev_rpy[0]) / dt
                pitch_rate = (pitch - data.prev_rpy[1]) / dt
                
                # Handle yaw wraparound (-pi to pi)
                yaw_diff = yaw - data.prev_rpy[2]
                if yaw_diff > math.pi:
                    yaw_diff -= 2 * math.pi
                elif yaw_diff < -math.pi:
                    yaw_diff += 2 * math.pi
                yaw_rate = yaw_diff / dt

                data.angular_vel = (roll_rate, pitch_rate, yaw_rate)

        data.prev_rpy = (roll, pitch, yaw)
        data.prev_rpy_time = current_time

    def publish_fused_twists(self):
        """Publish fused twist messages for all robots."""
        for robot_id in range(1, self.num_robots + 1):
            data = self.robot_data[robot_id]
            twist_msg = Twist()

            # Linear velocity from /vel
            if data.linear_vel is not None:
                twist_msg.linear.x = data.linear_vel[0]
                twist_msg.linear.y = data.linear_vel[1]
                twist_msg.linear.z = data.linear_vel[2]

            # Angular velocity from attitude derivative
            twist_msg.angular.x = data.angular_vel[0]
            twist_msg.angular.y = data.angular_vel[1]
            twist_msg.angular.z = data.angular_vel[2]

            self.twist_pubs[robot_id].publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
