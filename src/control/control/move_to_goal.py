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
        self.agent_ids = [1, 2, 3, 4, 7]

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

        # -------- State (keyed by agent ID) --------
        self.current_poses = {a: None for a in self.agent_ids}
        self.goal_poses = {a: None for a in self.agent_ids}
        self.last_pose_times = {a: None for a in self.agent_ids}

        # -------- Subscribers & Publishers --------
        self.pose_subs = {}
        self.goal_subs = {}
        self.cmd_vel_pubs = {}

        for robot_id in self.agent_ids:
            self.pose_subs[robot_id] = self.create_subscription(
                PoseStamped,
                f'/fused_pose_{robot_id}',
                lambda msg, rid=robot_id: self.pose_callback(msg, rid),
                10
            )

            self.goal_subs[robot_id] = self.create_subscription(
                Pose,
                f'/goal_pose_{robot_id}',
                lambda msg, rid=robot_id: self.goal_callback(msg, rid),
                10
            )

            # robomaster 1 & 2 use underscore, robomaster 7 does not
            cmd_vel_topic = {
                1: '/robomaster_1/cmd_vel',
                2: '/robomaster_2/cmd_vel',
                3: '/robomaster5/cmd_vel',
                4: '/robomaster4/cmd_vel',
                7: '/robomaster7/cmd_vel',
            }.get(robot_id, f'/robomaster_{robot_id}/cmd_vel')

            self.cmd_vel_pubs[robot_id] = self.create_publisher(
                Twist, cmd_vel_topic, 10
            )

        # Control loop timer
        self.control_timer = self.create_timer(control_period, self.control_callback)

        self.get_logger().info("🚗 MultiRobotGoalController started")
        self.get_logger().info(f"  Controlling agents {self.agent_ids}")
        self.get_logger().info(
            f"  Gains: kp_x={self.get_parameter('kp_x').value}, "
            f"kp_y={self.get_parameter('kp_y').value}, "
            f"kp_yaw={self.get_parameter('kp_yaw').value}"
        )

    # ---------- Callbacks ----------

    def pose_callback(self, msg: PoseStamped, robot_id: int):
        self.current_poses[robot_id] = msg.pose
        self.last_pose_times[robot_id] = self.get_clock().now()

    def goal_callback(self, msg: Pose, robot_id: int):
        self.goal_poses[robot_id] = msg
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

        for robot_id in self.agent_ids:
            pose = self.current_poses[robot_id]
            goal = self.goal_poses[robot_id]

            if pose is None or goal is None:
                self.cmd_vel_pubs[robot_id].publish(Twist())
                continue

            last_time = self.last_pose_times[robot_id]
            if last_time is not None:
                dt = (now - last_time).nanoseconds / 1e9
                if dt > 1.0:
                    self.cmd_vel_pubs[robot_id].publish(Twist())
                    continue

            cur_x = pose.position.x
            cur_y = pose.position.y
            cur_yaw = yaw_from_quat(pose.orientation)

            goal_x = goal.position.x
            goal_y = goal.position.y
            goal_yaw = yaw_from_quat(goal.orientation)

            err_x = goal_x - cur_x
            err_y = goal_y - cur_y
            err_yaw = wrap_angle(goal_yaw - cur_yaw)

            if math.hypot(err_x, err_y) < pos_tol and abs(err_yaw) < yaw_tol:
                self.cmd_vel_pubs[robot_id].publish(Twist())
                continue

            rot = np.array([
                [math.cos(cur_yaw), math.sin(cur_yaw)],
                [-math.sin(cur_yaw), math.cos(cur_yaw)]
            ])
            err_x_robot, err_y_robot = rot @ np.array([err_x, err_y])

            # Cardinal motion: only drive along the dominant error axis
            if abs(err_x_robot) >= abs(err_y_robot):
                vel_x_robot = kp_x * err_x_robot
                vel_y_robot = 0.0
            else:
                vel_x_robot = 0.0
                vel_y_robot = kp_y * err_y_robot
            vel_yaw = kp_yaw * err_yaw

            vel_x_robot = max(-max_linear_x, min(max_linear_x, vel_x_robot))
            vel_y_robot = max(-max_linear_y, min(max_linear_y, vel_y_robot))
            vel_yaw = max(-max_angular_z, min(max_angular_z, vel_yaw))

            twist = Twist()
            twist.linear.x = float(vel_x_robot)
            twist.linear.y = float(-vel_y_robot)
            twist.angular.z = float(-vel_yaw)

            self.cmd_vel_pubs[robot_id].publish(twist)

            self.get_logger().debug(
                f"[Robot {robot_id}] "
                f"err=(x={err_x:.3f}, y={err_y:.3f}, yaw={math.degrees(err_yaw):.1f}°), "
                f"cmd=(vx={vel_x_robot:.3f}, vy={vel_y_robot:.3f}, wz={math.degrees(vel_yaw):.1f}°/s)"
            )

    # ---------- Utility ----------

    def publish_stop_all(self):
        """Publish zero velocities to all robots."""
        twist = Twist()
        for pub in self.cmd_vel_pubs.values():
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()