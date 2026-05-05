#!/usr/bin/env python3
"""
ROS 2 node that subscribes to /fused_pose_* topics for all candidate agents,
and writes a timestamped CSV log of their positions for post-run analysis.

The CSV is written to the current working directory (or a configurable path)
with columns: timestamp, agent_id, x, y, z, yaw_deg
"""
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion


CANDIDATE_AGENT_IDS = [1, 2, 3, 4, 7]


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseLogger(Node):
    def __init__(self):
        super().__init__('pose_logger')

        self.declare_parameter('output_dir', '')
        self.declare_parameter('rate_hz', 10.0)

        output_dir = self.get_parameter('output_dir').value or os.getcwd()
        rate_hz = self.get_parameter('rate_hz').value

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(output_dir, f'pose_log_{timestamp}.csv')

        self._csv_file = open(self._csv_path, 'w', newline='')
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(['timestamp', 'agent_id', 'x', 'y', 'z', 'yaw_deg'])

        self._latest: dict[int, PoseStamped | None] = {}

        for agent_id in CANDIDATE_AGENT_IDS:
            self._latest[agent_id] = None
            self.create_subscription(
                PoseStamped,
                f'/fused_pose_{agent_id}',
                lambda msg, aid=agent_id: self._pose_cb(msg, aid),
                10,
            )

        self._log_timer = self.create_timer(1.0 / rate_hz, self._log_tick)

        self.get_logger().info(
            f'📝 PoseLogger started — writing to {self._csv_path} at {rate_hz} Hz'
        )

    def _pose_cb(self, msg: PoseStamped, agent_id: int):
        self._latest[agent_id] = msg

    def _log_tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        for agent_id, msg in self._latest.items():
            if msg is None:
                continue
            p = msg.pose
            yaw_deg = math.degrees(yaw_from_quat(p.orientation))
            self._writer.writerow([
                f'{now:.4f}',
                agent_id,
                f'{p.position.x:.4f}',
                f'{p.position.y:.4f}',
                f'{p.position.z:.4f}',
                f'{yaw_deg:.2f}',
            ])
        self._csv_file.flush()

    def destroy_node(self):
        self._csv_file.close()
        self.get_logger().info(f'📝 PoseLogger closed — log saved to {self._csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseLogger()

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
