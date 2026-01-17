#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class Camera0Publisher(Node):
    def __init__(self):
        super().__init__('camera0_publisher')

        # Publisher
        self.pub = self.create_publisher(Image, '/camera0/image_raw', 10)

        # cv_bridge for converting OpenCV → ROS Image
        self.bridge = CvBridge()

        # Open webcam 0
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Could not open webcam 0.")
            raise RuntimeError("Webcam 0 not accessible.")

        self.get_logger().info("📷 Camera0 publisher started on /camera0/image_raw")

        # Timer for publishing at ~30 Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("⚠️ Failed to read frame from webcam 0.")
            return

        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Publish
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Camera0Publisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()