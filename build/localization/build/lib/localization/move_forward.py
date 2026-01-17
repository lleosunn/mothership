#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
import time


class MoveForwardNode(Node):
    def __init__(self):
        super().__init__('move_forward')

        # QoS profile to match the robot (best effort reliability)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher = self.create_publisher(
            Twist,
            '/robomaster_1/cmd_vel',
            qos_profile
        )

        self.get_logger().info("🚀 Move Forward Node started.")
        self.get_logger().info("   Publishing to /robomaster_1/cmd_vel")

        # Start the movement sequence
        self.execute_movement()

    def execute_movement(self):
        # Create forward velocity command
        forward_cmd = Twist()
        forward_cmd.linear.x = 1.0
        forward_cmd.linear.y = 0.0
        forward_cmd.linear.z = 0.0
        forward_cmd.angular.x = 0.0
        forward_cmd.angular.y = 0.0
        forward_cmd.angular.z = 0.0

        # Create stop command
        stop_cmd = Twist()

        # Move forward for 1 second
        self.get_logger().info("Moving forward at 1.0 m/s...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            self.publisher.publish(forward_cmd)
            time.sleep(0.05)  # Publish at ~20 Hz

        # Stop - loop forever
        self.get_logger().info("Stopping (Ctrl+C to exit)...")
        self.stop_cmd = stop_cmd

    def spin_stop(self):
        """Continuously publish stop command."""
        while rclpy.ok():
            self.publisher.publish(self.stop_cmd)
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = MoveForwardNode()

    try:
        node.spin_stop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
