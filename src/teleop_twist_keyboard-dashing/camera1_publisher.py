#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class Camera1Publisher(Node):
    def __init__(self):
        super().__init__('camera1_publisher')

        # Publisher
        self.pub = self.create_publisher(Image, '/camera1/image_raw', 10)

        # cv_bridge for converting OpenCV → ROS Image
        self.bridge = CvBridge()

        # Open webcam 1
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Could not open webcam 1.")
            raise RuntimeError("Webcam 1 not accessible.")

        self.get_logger().info("📷 Camera1 publisher started on /camera1/image_raw")

        # Timer for publishing at ~30 Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("⚠️ Failed to read frame from webcam 1.")
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
    node = Camera1Publisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()