#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import math
import numpy as np
import sys
import threading

class RepulsiveAvoidanceNode(Node):
    def __init__(self):
        super().__init__('aruco_controller_node')
        self.num_agents = 2
        self.pose_subs = []
        self.current_poses = [None] * self.num_agents
        for i in range(self.num_agents):
            self.pose_subs.append(
                self.create_subscription(
                    Pose,
                    f'/fused_pose_{i+1}',
                    lambda msg, idx=i: self.pose_callback(msg, idx),
                    10
                )
            )

        # --- Declare parameters (editable via command line or parameter file) ---
        # Define default coordinates and yaws here (modifiable in code)
        # (x, y, yaw_degrees): target at (0, 0.5) faces 90°, target at (1, 0.5) faces -90°
        self.default_targets = [(0, 0.5, 90.0), (1, 0.5, -90.0)]

        self.target_positions = []
        self.target_yaws = []
        for i in range(self.num_agents):
            default_x, default_y, default_yaw = self.default_targets[i] if i < len(self.default_targets) else (0.5, 0.5, 0.0)
            self.declare_parameter(f'target_x_{i+1}', default_x)
            self.declare_parameter(f'target_y_{i+1}', default_y)
            self.declare_parameter(f'target_yaw_{i+1}', default_yaw)
            self.target_positions.append((
                self.get_parameter(f'target_x_{i+1}').value,
                self.get_parameter(f'target_y_{i+1}').value
            ))
            self.target_yaws.append(math.radians(self.get_parameter(f'target_yaw_{i+1}').value))
        
        # P controller gains
        self.declare_parameter('kp_x', 1)  # Proportional gain for x
        self.declare_parameter('kp_y', 2)  # Proportional gain for y
        self.declare_parameter('kp_yaw', 1)  # Proportional gain for yaw
        
        # Maximum velocities (m/s)
        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('max_linear_y', 0.25)
        self.declare_parameter('max_angular_z', 1.0)  # Maximum angular velocity (rad/s)
        
        # Position tolerance (meters) - stop when within this distance
        self.declare_parameter('position_tolerance', 0.05)

        # Repulsive avoidance parameters (no time memory)
        self.declare_parameter('repulsion_distance', 1.0)   # meters
        self.declare_parameter('repulsion_gain', 1.5)        # magnitude of repulsive term
        
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
            tyaw = math.degrees(self.target_yaws[i])
            self.get_logger().info(f"  Target for agent {i+1}: x={tx:.3f}, y={ty:.3f}, yaw={tyaw:.1f}°")
        self.get_logger().info(f"  P gains: kp_x={self.get_parameter('kp_x').value:.3f}, kp_y={self.get_parameter('kp_y').value:.3f}")

        # --- Keyboard listener thread for swapping targets ---
        self.get_logger().info("⌨️  Press Enter in this terminal to swap targets between the two agents.")
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

    def pose_callback(self, msg, agent_index):
        """Update current pose from ArUco detection."""
        self.current_poses[agent_index] = msg
        if agent_index == 0:
            self.last_pose_time = self.get_clock().now()

    def keyboard_listener(self):
        """
        Simple blocking keyboard listener running in a background thread.
        Every time the user hits Enter in the terminal where this node is running,
        the targets for agent 1 and agent 2 are swapped.
        """
        # We only support swapping between two agents here
        if self.num_agents < 2:
            return

        for line in sys.stdin:
            # Any line (i.e., pressing Enter) triggers a swap
            self.swap_targets()

    def swap_targets(self):
        """Swap the target positions and yaws of agent 1 and agent 2."""
        if self.num_agents < 2:
            return
        self.target_positions[0], self.target_positions[1] = (
            self.target_positions[1],
            self.target_positions[0],
        )
        self.target_yaws[0], self.target_yaws[1] = (
            self.target_yaws[1],
            self.target_yaws[0],
        )
        (tx1, ty1) = self.target_positions[0]
        (tx2, ty2) = self.target_positions[1]
        tyaw1 = math.degrees(self.target_yaws[0])
        tyaw2 = math.degrees(self.target_yaws[1])
        self.get_logger().info(
            f"🔁 Swapped targets: agent 1 -> (x={tx1:.3f}, y={ty1:.3f}, yaw={tyaw1:.1f}°), "
            f"agent 2 -> (x={tx2:.3f}, y={ty2:.3f}, yaw={tyaw2:.1f}°)"
        )

    def control_callback(self):
        """Main control loop - calculates and publishes velocity commands for all agents."""
        # Get gains and limits from parameters
        kp_x = self.get_parameter('kp_x').value
        kp_y = self.get_parameter('kp_y').value
        kp_yaw = self.get_parameter('kp_yaw').value
        max_linear_x = self.get_parameter('max_linear_x').value
        max_linear_y = self.get_parameter('max_linear_y').value
        max_angular_z = self.get_parameter('max_angular_z').value
        tolerance = self.get_parameter('position_tolerance').value
        repulsion_distance = self.get_parameter('repulsion_distance').value
        repulsion_gain = self.get_parameter('repulsion_gain').value

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
            # Get per-agent target (position and yaw)
            target_x, target_y = self.target_positions[agent_index]
            target_yaw = self.target_yaws[agent_index]
            current_x = pose.position.x
            current_y = pose.position.y

            # --- Pure distance-based repulsive avoidance between agents ---
            repulse_x = 0.0
            repulse_y = 0.0
            for other_index in range(self.num_agents):
                if other_index == agent_index:
                    continue
                other_pose = self.current_poses[other_index]
                if other_pose is None:
                    continue
                dx = current_x - other_pose.position.x
                dy = current_y - other_pose.position.y
                dist = math.hypot(dx, dy)
                if dist < repulsion_distance and dist > 1e-6:
                    # Unit vector pointing away from the other agent
                    ux = dx / dist
                    uy = dy / dist
                    # Stronger repulsion the closer they are (linear in gap)
                    strength = repulsion_gain * (repulsion_distance - dist) / repulsion_distance
                    repulse_x += strength * ux
                    repulse_y += strength * uy

            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            current_yaw = math.atan2(siny_cosp, cosy_cosp)
            # Attractive term towards goal plus instantaneous repulsive term to avoid other robots
            error_x = (target_x - current_x) + repulse_x
            error_y = (target_y - current_y) + repulse_y
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
    node = RepulsiveAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()