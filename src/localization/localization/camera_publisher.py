#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('device_index', 0)

        camera_id = self.get_parameter('camera_id').value
        device_index = self.get_parameter('device_index').value

        topic = f'/camera{camera_id}/image_raw'
        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"❌ Could not open webcam (camera_id={camera_id}, device_index={device_index})."
            )
            raise RuntimeError(
                f"Webcam device {device_index} not accessible."
            )

        self.get_logger().info(
            f"📷 Camera{camera_id} publisher started on {topic} "
            f"(device_index={device_index})"
        )

        self.timer = self.create_timer(1 / 15.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("⚠️ Failed to read frame.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        msg = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()

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


if __name__ == '__main__':
    main()
