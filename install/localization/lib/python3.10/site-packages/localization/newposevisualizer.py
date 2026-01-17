#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
import numpy as np
import time
import math


def quaternion_to_yaw(q):
    """Extract yaw angle from quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class PoseVisualizerNode(Node):
    def __init__(self):
        super().__init__('pose_visualizer')

        # Store latest poses (x, y, yaw) and timestamps
        self.pose1 = None
        self.pose2 = None
        self.pose1_last_time = 0.0
        self.pose2_last_time = 0.0

        # Timeout in seconds
        self.pose_timeout = 0.5

        # Arrow length for visualization
        self.arrow_length = 0.08

        # Subscribers for fused poses
        self.pose1_sub = self.create_subscription(
            Pose, '/pose_1', self.pose1_callback, 10)
        self.pose2_sub = self.create_subscription(
            Pose, '/pose_2', self.pose2_callback, 10)

        # Setup matplotlib figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle('Real-time Pose Visualization', fontsize=14, fontweight='bold')

        # Scatter plots for agent positions (single points, updated via set_offsets)
        self.scatter_pose1 = self.ax.scatter([], [], c='#00FFFF', s=200, label='Agent 1',
                                              marker='o', edgecolors='white', linewidths=2, zorder=5)
        self.scatter_pose2 = self.ax.scatter([], [], c='#50FA7B', s=200, label='Agent 2',
                                              marker='o', edgecolors='white', linewidths=2, zorder=5)

        # Line segments for orientation (much faster than quiver)
        self.line_pose1, = self.ax.plot([], [], color='#00FFFF', linewidth=3, solid_capstyle='round', zorder=4)
        self.line_pose2, = self.ax.plot([], [], color='#50FA7B', linewidth=3, solid_capstyle='round', zorder=4)

        # Configure axes
        self.ax.set_xlim(-0.5, 1)
        self.ax.set_ylim(-0.5, 1)
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper right', fontsize=10)

        # Text annotations for coordinates
        self.text_pose1 = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top', color='#00FFFF',
                                        fontfamily='monospace')
        self.text_pose2 = self.ax.text(0.02, 0.93, '', transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top', color='#50FA7B',
                                        fontfamily='monospace')

        self.get_logger().info("📊 Pose Visualizer Node started.")
        self.get_logger().info("   Subscribing to /pose_1 and /pose_2")

    def extract_pose(self, msg):
        """Extract (x, y, yaw) from Pose message."""
        x = msg.position.x
        y = msg.position.y
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        yaw = quaternion_to_yaw(q)
        return (x, y, yaw)

    def pose1_callback(self, msg):
        self.pose1 = self.extract_pose(msg)
        self.pose1_last_time = time.time()

    def pose2_callback(self, msg):
        self.pose2 = self.extract_pose(msg)
        self.pose2_last_time = time.time()

    def update_plot(self):
        current_time = time.time()

        # === AGENT 1 ===
        pose1_valid = (self.pose1 is not None and
                       (current_time - self.pose1_last_time) < self.pose_timeout)

        if pose1_valid:
            x, y, yaw = self.pose1
            self.scatter_pose1.set_offsets([(x, y)])
            # Line segment from position to arrow tip
            x2 = x + self.arrow_length * math.cos(yaw)
            y2 = y + self.arrow_length * math.sin(yaw)
            self.line_pose1.set_data([x, x2], [y, y2])
            self.text_pose1.set_text(f'Agent1: x={x:.3f}, y={y:.3f}, θ={math.degrees(yaw):.1f}°')
        else:
            self.scatter_pose1.set_offsets(np.empty((0, 2)))
            self.line_pose1.set_data([], [])
            self.text_pose1.set_text('Agent1: --')

        # === AGENT 2 ===
        pose2_valid = (self.pose2 is not None and
                       (current_time - self.pose2_last_time) < self.pose_timeout)

        if pose2_valid:
            x, y, yaw = self.pose2
            self.scatter_pose2.set_offsets([(x, y)])
            x2 = x + self.arrow_length * math.cos(yaw)
            y2 = y + self.arrow_length * math.sin(yaw)
            self.line_pose2.set_data([x, x2], [y, y2])
            self.text_pose2.set_text(f'Agent2: x={x:.3f}, y={y:.3f}, θ={math.degrees(yaw):.1f}°')
        else:
            self.scatter_pose2.set_offsets(np.empty((0, 2)))
            self.line_pose2.set_data([], [])
            self.text_pose2.set_text('Agent2: --')

    def run(self):
        """Run the node with matplotlib integration."""
        plt.ion()
        plt.show(block=False)

        while rclpy.ok():
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.001)
            
            # Update plot
            self.update_plot()
            
            # Minimal pause to update display
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = PoseVisualizerNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
