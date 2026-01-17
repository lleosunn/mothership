#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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

        self.num_agents = 2

        # Store latest poses (x, y, yaw) and timestamps for raw and fused
        self.raw_poses = {i: None for i in range(1, self.num_agents + 1)}
        self.raw_times = {i: 0.0 for i in range(1, self.num_agents + 1)}
        self.fused_poses = {i: None for i in range(1, self.num_agents + 1)}
        self.fused_times = {i: 0.0 for i in range(1, self.num_agents + 1)}

        # Timeout in seconds
        self.pose_timeout = 0.5

        # Agent radius and arrow length for visualization
        self.agent_radius = 0.15
        self.arrow_length = 0.2

        # Colors for each agent (raw is semi-transparent, fused is solid)
        self.agent_colors = {
            1: {'raw': '#00FFFF', 'fused': '#00FFFF'},  # Cyan
            2: {'raw': '#50FA7B', 'fused': '#50FA7B'},  # Green
        }

        # Subscribers for raw poses
        self.raw_subs = []
        for i in range(1, self.num_agents + 1):
            sub = self.create_subscription(
                Pose, f'/pose_{i}',
                lambda msg, agent_id=i: self.raw_callback(msg, agent_id),
                10
            )
            self.raw_subs.append(sub)

        # Subscribers for fused poses
        self.fused_subs = []
        for i in range(1, self.num_agents + 1):
            sub = self.create_subscription(
                Pose, f'/fused_pose_{i}',
                lambda msg, agent_id=i: self.fused_callback(msg, agent_id),
                10
            )
            self.fused_subs.append(sub)

        # Setup matplotlib figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.suptitle('Raw vs Fused Pose Visualization', fontsize=14, fontweight='bold')

        # Create circle patches and line segments for each agent (raw + fused)
        self.raw_circles = {}
        self.raw_lines = {}
        self.fused_circles = {}
        self.fused_lines = {}

        for i in range(1, self.num_agents + 1):
            color = self.agent_colors[i]['raw']
            
            # Raw pose - dashed edge, semi-transparent
            raw_circle = Circle((0, 0), self.agent_radius, 
                                 facecolor=color, alpha=0.3,
                                 edgecolor=color, linewidth=2, linestyle='--',
                                 zorder=4, label=f'Agent {i} Raw')
            self.ax.add_patch(raw_circle)
            raw_circle.set_visible(False)
            self.raw_circles[i] = raw_circle
            
            raw_line, = self.ax.plot([], [], color=color, linewidth=2, 
                                      linestyle='--', alpha=0.5, zorder=4)
            self.raw_lines[i] = raw_line

            # Fused pose - solid edge, full opacity
            fused_circle = Circle((0, 0), self.agent_radius,
                                   facecolor=color, alpha=0.8,
                                   edgecolor='white', linewidth=2,
                                   zorder=5, label=f'Agent {i} Fused')
            self.ax.add_patch(fused_circle)
            fused_circle.set_visible(False)
            self.fused_circles[i] = fused_circle
            
            fused_line, = self.ax.plot([], [], color='white', linewidth=3,
                                        solid_capstyle='round', zorder=6)
            self.fused_lines[i] = fused_line

        # Configure axes
        self.ax.set_xlim(-1.5, 2.5)
        self.ax.set_ylim(-1.5, 2.5)
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper right', fontsize=9)

        # Text annotations for coordinates
        self.text_labels = {}
        y_offset = 0.98
        for i in range(1, self.num_agents + 1):
            color = self.agent_colors[i]['raw']
            self.text_labels[f'raw_{i}'] = self.ax.text(
                0.02, y_offset, '', transform=self.ax.transAxes,
                fontsize=9, verticalalignment='top', color=color,
                fontfamily='monospace', alpha=0.7
            )
            y_offset -= 0.04
            self.text_labels[f'fused_{i}'] = self.ax.text(
                0.02, y_offset, '', transform=self.ax.transAxes,
                fontsize=9, verticalalignment='top', color=color,
                fontfamily='monospace', fontweight='bold'
            )
            y_offset -= 0.05

        self.get_logger().info("📊 Pose Visualizer Node started.")
        self.get_logger().info("   Subscribing to /pose_N and /fused_pose_N for each agent")

    def extract_pose(self, msg):
        """Extract (x, y, yaw) from Pose message."""
        x = msg.position.x
        y = msg.position.y
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        yaw = quaternion_to_yaw(q)
        return (x, y, yaw)

    def raw_callback(self, msg, agent_id):
        self.raw_poses[agent_id] = self.extract_pose(msg)
        self.raw_times[agent_id] = time.time()

    def fused_callback(self, msg, agent_id):
        self.fused_poses[agent_id] = self.extract_pose(msg)
        self.fused_times[agent_id] = time.time()

    def update_plot(self):
        current_time = time.time()

        for i in range(1, self.num_agents + 1):
            # === RAW POSE ===
            raw_valid = (self.raw_poses[i] is not None and
                         (current_time - self.raw_times[i]) < self.pose_timeout)

            if raw_valid:
                x, y, yaw = self.raw_poses[i]
                self.raw_circles[i].set_center((x, y))
                self.raw_circles[i].set_visible(True)
                x2 = x + self.arrow_length * math.cos(yaw)
                y2 = y + self.arrow_length * math.sin(yaw)
                self.raw_lines[i].set_data([x, x2], [y, y2])
                self.text_labels[f'raw_{i}'].set_text(
                    f'Agent{i} Raw:   x={x:.3f}, y={y:.3f}, θ={math.degrees(yaw):.1f}°'
                )
            else:
                self.raw_circles[i].set_visible(False)
                self.raw_lines[i].set_data([], [])
                self.text_labels[f'raw_{i}'].set_text(f'Agent{i} Raw:   --')

            # === FUSED POSE ===
            fused_valid = (self.fused_poses[i] is not None and
                           (current_time - self.fused_times[i]) < self.pose_timeout)

            if fused_valid:
                x, y, yaw = self.fused_poses[i]
                self.fused_circles[i].set_center((x, y))
                self.fused_circles[i].set_visible(True)
                x2 = x + self.arrow_length * math.cos(yaw)
                y2 = y + self.arrow_length * math.sin(yaw)
                self.fused_lines[i].set_data([x, x2], [y, y2])
                self.text_labels[f'fused_{i}'].set_text(
                    f'Agent{i} Fused: x={x:.3f}, y={y:.3f}, θ={math.degrees(yaw):.1f}°'
                )
            else:
                self.fused_circles[i].set_visible(False)
                self.fused_lines[i].set_data([], [])
                self.text_labels[f'fused_{i}'].set_text(f'Agent{i} Fused: --')

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
