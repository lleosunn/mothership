#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Twist


def yaw_from_quat(q):
    """Extract yaw from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MultiRobotGoalController(Node):
    def __init__(self):
        super().__init__('multi_robot_goal_controller')

        # -------- Parameters --------
        # Number of robots
        self.num_agents = 2

        # P gains
        self.declare_parameter('kp_x', 2.0)
        self.declare_parameter('kp_y', 2.0)
        self.declare_parameter('kp_yaw', 1.0)

        # Velocity limits
        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('max_linear_y', 0.25)
        self.declare_parameter('max_angular_z', 1.0)

        # Position & yaw tolerance
        self.declare_parameter('position_tolerance', 0.05)
        self.declare_parameter('yaw_tolerance', 0.05)  # radians

        # Control rate
        self.declare_parameter('control_rate', 20.0)
        control_rate = float(self.get_parameter('control_rate').value)
        control_period = 1.0 / control_rate

        # -------- State --------
        # Current and goal poses for each robot
        self.current_poses = [None] * self.num_agents
        self.goal_poses = [None] * self.num_agents
        self.last_pose_times = [None] * self.num_agents

        # -------- Subscribers & Publishers --------
        self.pose_subs = []
        self.goal_subs = []
        self.cmd_vel_pubs = []

        for robot_id in range(1, self.num_agents + 1):
            idx = robot_id - 1

            pose_sub = self.create_subscription(
                PoseStamped,
                f'/fused_pose_{robot_id}',
                lambda msg, i=idx: self.pose_callback(msg, i),
                10
            )
            self.pose_subs.append(pose_sub)

            # Goal pose
            goal_sub = self.create_subscription(
                Pose,
                f'/goal_pose_{robot_id}',
                lambda msg, i=idx: self.goal_callback(msg, i),
                10
            )
            self.goal_subs.append(goal_sub)

            # Velocity command publisher
            cmd_pub = self.create_publisher(
                Twist,
                f'/robomaster_{robot_id}/cmd_vel',
                10
            )
            self.cmd_vel_pubs.append(cmd_pub)

        # Control loop timer
        self.control_timer = self.create_timer(control_period, self.control_callback)

        self.get_logger().info("🚗 MultiRobotGoalController started")
        self.get_logger().info(f"  Controlling {self.num_agents} agents")
        self.get_logger().info(
            f"  Gains: kp_x={self.get_parameter('kp_x').value}, "
            f"kp_y={self.get_parameter('kp_y').value}, "
            f"kp_yaw={self.get_parameter('kp_yaw').value}"
        )

    # ---------- Callbacks ----------

    def pose_callback(self, msg: PoseStamped, idx: int):
        """Update current fused pose for robot idx."""
        self.current_poses[idx] = msg.pose
        self.last_pose_times[idx] = self.get_clock().now()

    def goal_callback(self, msg: Pose, idx: int):
        """Update goal pose for robot idx."""
        self.goal_poses[idx] = msg
        # self.get_logger().info(
        #     f"[Robot {idx+1}] New goal: "
        #     f"x={msg.position.x:.3f}, y={msg.position.y:.3f}"
        # )

    # ---------- Control loop ----------

    def control_callback(self):
        """Main control loop for all robots."""
        # Read parameters (so you can tune at runtime)
        kp_x = float(self.get_parameter('kp_x').value)
        kp_y = float(self.get_parameter('kp_y').value)
        kp_yaw = float(self.get_parameter('kp_yaw').value)

        max_linear_x = float(self.get_parameter('max_linear_x').value)
        max_linear_y = float(self.get_parameter('max_linear_y').value)
        max_angular_z = float(self.get_parameter('max_angular_z').value)

        pos_tol = float(self.get_parameter('position_tolerance').value)
        yaw_tol = float(self.get_parameter('yaw_tolerance').value)

        now = self.get_clock().now()

        for idx in range(self.num_agents):
            pose = self.current_poses[idx]
            goal = self.goal_poses[idx]

            # If no pose or goal, stop this robot
            if pose is None or goal is None:
                self.cmd_vel_pubs[idx].publish(Twist())
                continue

            # Stop robot if pose info is stale
            last_time = self.last_pose_times[idx]
            if last_time is not None:
                dt = (now - last_time).nanoseconds / 1e9
                if dt > 1.0:
                    self.cmd_vel_pubs[idx].publish(Twist())
                    continue

            # Extract current pose
            cur_x = pose.position.x
            cur_y = pose.position.y
            cur_yaw = yaw_from_quat(pose.orientation)

            # Extract goal pose
            goal_x = goal.position.x
            goal_y = goal.position.y
            goal_yaw = yaw_from_quat(goal.orientation)

            # Position error in global frame
            err_x = goal_x - cur_x
            err_y = goal_y - cur_y

            # Yaw error
            err_yaw = wrap_angle(goal_yaw - cur_yaw)

            # Check if within tolerance → stop
            if math.hypot(err_x, err_y) < pos_tol and abs(err_yaw) < yaw_tol:
                self.cmd_vel_pubs[idx].publish(Twist())
                continue

            # Transform position error into robot frame
            # Same convention as your original node:
            # [err_x_robot, err_y_robot] = R * [err_x, err_y]
            rot = np.array([
                [math.cos(cur_yaw), math.sin(cur_yaw)],
                [-math.sin(cur_yaw), math.cos(cur_yaw)]
            ])
            err_x_robot, err_y_robot = rot @ np.array([err_x, err_y])

            # P-control in robot frame
            vel_x_robot = kp_x * err_x_robot
            vel_y_robot = kp_y * err_y_robot
            vel_yaw = kp_yaw * err_yaw

            # Clamp velocities
            vel_x_robot = max(-max_linear_x, min(max_linear_x, vel_x_robot))
            vel_y_robot = max(-max_linear_y, min(max_linear_y, vel_y_robot))
            vel_yaw = max(-max_angular_z, min(max_angular_z, vel_yaw))

            # Build Twist (matching your sign conventions)
            twist = Twist()
            twist.linear.x = float(vel_x_robot)
            twist.linear.y = float(-vel_y_robot)
            twist.angular.z = float(-vel_yaw)

            self.cmd_vel_pubs[idx].publish(twist)

            # Optional debug log (can be noisy)
            self.get_logger().debug(
                f"[Robot {idx+1}] "
                f"err=(x={err_x:.3f}, y={err_y:.3f}, yaw={math.degrees(err_yaw):.1f}°), "
                f"cmd=(vx={vel_x_robot:.3f}, vy={vel_y_robot:.3f}, wz={math.degrees(vel_yaw):.1f}°/s)"
            )

    # ---------- Utility ----------

    def publish_stop_all(self):
        """Publish zero velocities to all robots."""
        twist = Twist()
        for pub in self.cmd_vel_pubs:
            pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotGoalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()