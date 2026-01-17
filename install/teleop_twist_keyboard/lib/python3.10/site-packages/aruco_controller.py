#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import math
import numpy as np

class ArucoControllerNode(Node):
    def __init__(self):
        super().__init__('aruco_controller_node')
        self.num_agents = 2
        self.pose_subs = []
        self.current_poses = [None] * self.num_agents
        for i in range(self.num_agents):
            self.pose_subs.append(
                self.create_subscription(
                    Pose,
                    f'/pose_{i+1}',
                    lambda msg, idx=i: self.pose_callback(msg, idx),
                    10
                )
            )

        # --- Declare parameters (editable via command line or parameter file) ---
        # Define default coordinates here (modifiable in code)
        self.default_targets = [(0.4, 0.4), (0.8, 0.8)]

        self.target_positions = []
        for i in range(self.num_agents):
            default_x, default_y = self.default_targets[i] if i < len(self.default_targets) else (0.5, 0.5)
            self.declare_parameter(f'target_x_{i+1}', default_x)
            self.declare_parameter(f'target_y_{i+1}', default_y)
            self.target_positions.append((
                self.get_parameter(f'target_x_{i+1}').value,
                self.get_parameter(f'target_y_{i+1}').value
            ))
        
        # P controller gains
        self.declare_parameter('kp_x', 1)  # Proportional gain for x
        self.declare_parameter('kp_y', 2)  # Proportional gain for y
        self.declare_parameter('kp_yaw', 1)  # Proportional gain for yaw
        
        # Maximum velocities (m/s)
        self.declare_parameter('max_linear_x', 0.5)
        self.declare_parameter('max_linear_y', 0.5)
        self.declare_parameter('max_angular_z', 1.0)  # Maximum angular velocity (rad/s)
        
        # Position tolerance (meters) - stop when within this distance
        self.declare_parameter('position_tolerance', 0.05)
        
        # Control rate (Hz)
        self.declare_parameter('control_rate', 20.0)

        # --- Publishers ---
        self.cmd_vel_pubs = []
        for i in range(self.num_agents):
            self.cmd_vel_pubs.append(
                self.create_publisher(
                    Twist,
                    f'/robomaster_{i+1}/cmd_vel',
                    10
                )
            )

        # Current poses (will be updated by subscription)
        self.last_pose_time = None

        # --- Control timer ---
        control_period = 1.0 / self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(control_period, self.control_callback)

        self.get_logger().info("📡 ArUco Controller Node started")
        for i, (tx, ty) in enumerate(self.target_positions):
            self.get_logger().info(f"  Target position for agent {i+1}: x={tx:.3f}, y={ty:.3f} (default {self.default_targets[i]})")
        self.get_logger().info(f"  P gains: kp_x={self.get_parameter('kp_x').value:.3f}, kp_y={self.get_parameter('kp_y').value:.3f}")

    def pose_callback(self, msg, agent_index):
        """Update current pose from ArUco detection."""
        self.current_poses[agent_index] = msg
        if agent_index == 0:
            self.last_pose_time = self.get_clock().now()

    def control_callback(self):
        """Main control loop - calculates and publishes velocity commands for all agents."""
        # Get gains and limits from parameters
        target_yaw = 0.0  # Always target 0 yaw
        kp_x = self.get_parameter('kp_x').value
        kp_y = self.get_parameter('kp_y').value
        kp_yaw = self.get_parameter('kp_yaw').value
        max_linear_x = self.get_parameter('max_linear_x').value
        max_linear_y = self.get_parameter('max_linear_y').value
        max_angular_z = self.get_parameter('max_angular_z').value
        tolerance = self.get_parameter('position_tolerance').value

        for agent_index in range(self.num_agents):
            pose = self.current_poses[agent_index]
            if self.last_pose_time is not None:
                time_since_pose = (self.get_clock().now() - self.last_pose_time).nanoseconds / 1e9
                if time_since_pose > 1.0:
                    self.cmd_vel_pubs[agent_index].publish(Twist())  # stop agent
                    continue
            if pose is None:
                self.cmd_vel_pubs[agent_index].publish(Twist())  # stop this agent
                continue
            # Get per-agent target
            target_x, target_y = self.target_positions[agent_index]
            current_x = pose.position.x
            current_y = pose.position.y
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            current_yaw = math.atan2(siny_cosp, cosy_cosp)
            error_x = target_x - current_x
            error_y = target_y - current_y
            error_yaw = target_yaw - current_yaw
            while error_yaw > math.pi:
                error_yaw -= 2 * math.pi
            while error_yaw < -math.pi:
                error_yaw += 2 * math.pi
            rotation_matrix = np.array([[math.cos(current_yaw), math.sin(current_yaw)],
                                        [-math.sin(current_yaw), math.cos(current_yaw)]])
            error_xrobot, error_yrobot = rotation_matrix @ np.array([error_x, error_y])
            vel_xrobot = kp_x * error_xrobot
            vel_yrobot = kp_y * error_yrobot
            vel_yaw = kp_yaw * error_yaw
            vel_xrobot = max(-max_linear_x, min(max_linear_x, vel_xrobot))
            vel_yrobot = max(-max_linear_y, min(max_linear_y, vel_yrobot))
            vel_yaw = max(-max_angular_z, min(max_angular_z, vel_yaw))
            twist = Twist()
            twist.linear.x = float(vel_xrobot)
            twist.linear.y = float(-vel_yrobot)
            twist.angular.z = float(-vel_yaw)
            self.cmd_vel_pubs[agent_index].publish(twist)
            self.get_logger().info(f"Agent {agent_index+1} Vel: ({vel_xrobot:.3f}, {vel_yrobot:.3f}, {math.degrees(vel_yaw):.1f}°/s)")

    def publish_stop(self):
        """Publish zero velocity to stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        for pub in self.cmd_vel_pubs:
            pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()