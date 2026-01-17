#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import QuaternionStamped
import math


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (in degrees)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class AttitudeToRPYNode(Node):
    def __init__(self):
        super().__init__('attitude_to_rpy')

        # QoS profile to match the publisher (best effort reliability)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            QuaternionStamped,
            '/robomaster_1/attitude',
            self.attitude_callback,
            qos_profile
        )

        self.get_logger().info("🎯 Attitude to RPY Node started.")
        self.get_logger().info("   Subscribing to /robomaster_1/attitude")

    def attitude_callback(self, msg):
        q = msg.quaternion
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        
        self.get_logger().info(
            f"Roll: {roll:7.2f}°  Pitch: {pitch:7.2f}°  Yaw: {yaw:7.2f}°"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AttitudeToRPYNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
