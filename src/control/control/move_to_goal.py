#!/usr/bin/env python3
from typing import Optional

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


def _as_bool(value) -> bool:
    """ROS launch XML often passes booleans as strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


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

        # If > 0: when position error is below this (m), stop cmd_vel even if yaw is still
        # outside yaw_tolerance. Use for swap / holonomic goals where final heading is
        # hard to nail exactly (avoids robots hunting forever and colliding).
        self.declare_parameter('stop_if_position_within_m', 0.0)

        # If True: only track goal (x,y); match goal yaw to current fused yaw so /goal_pose
        # yaw=0 from cbs_scenario does not command huge spins. "At goal" uses position only.
        self.declare_parameter('position_only_goals', True)

        # Stop cmd_vel if /fused_pose has not updated for this long (vision can be bursty).
        self.declare_parameter('pose_stale_timeout_s', 3.0)

        # Control rate
        self.declare_parameter('control_rate', 20.0)
        control_rate = float(self.get_parameter('control_rate').value)
        control_period = 1.0 / control_rate

        # Verbose per-tick logs (usually off; use publish_debug_poses for RViz / ros2 topic echo)
        self.declare_parameter('explain_motion', False)
        self.declare_parameter('explain_motion_period_s', 0.4)

        # PoseStamped topics for comparing fused estimate vs active /goal_pose (RViz: add both).
        self.declare_parameter('publish_debug_poses', True)
        self.declare_parameter('debug_frame_id', 'world')

        # -------- State --------
        # Current and goal poses for each robot
        self.current_poses = [None] * self.num_agents
        self.goal_poses = [None] * self.num_agents
        self.last_pose_times = [None] * self.num_agents
        self._last_explain_ns = [0] * self.num_agents
        self._last_fused_stamped = [None] * self.num_agents

        # -------- Subscribers & Publishers --------
        self.pose_subs = []
        self.goal_subs = []
        self.cmd_vel_pubs = []
        self._debug_fused_pubs = []
        self._debug_goal_pubs = []

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

            self._debug_fused_pubs.append(
                self.create_publisher(
                    PoseStamped,
                    f'/move_to_goal/debug/robot_{robot_id}/fused_pose',
                    10,
                )
            )
            self._debug_goal_pubs.append(
                self.create_publisher(
                    PoseStamped,
                    f'/move_to_goal/debug/robot_{robot_id}/goal_pose',
                    10,
                )
            )

        # Control loop timer
        self.control_timer = self.create_timer(control_period, self.control_callback)

        self.get_logger().info("🚗 MultiRobotGoalController started")
        self.get_logger().info(f"  Controlling {self.num_agents} agents")
        self.get_logger().info(
            f"  Gains: kp_x={self.get_parameter('kp_x').value}, "
            f"kp_y={self.get_parameter('kp_y').value}, "
            f"kp_yaw={self.get_parameter('kp_yaw').value}"
        )
        self.get_logger().info(
            f"  explain_motion={self.get_parameter('explain_motion').value}, "
            f"period={self.get_parameter('explain_motion_period_s').value}s, "
            f"position_only_goals={self.get_parameter('position_only_goals').value}, "
            f"pose_stale_timeout_s={self.get_parameter('pose_stale_timeout_s').value}, "
            f"publish_debug_poses={self.get_parameter('publish_debug_poses').value}, "
            f"stop_if_position_within_m={self.get_parameter('stop_if_position_within_m').value}"
        )
        self.get_logger().info(
            '  Debug PoseStamped: /move_to_goal/debug/robot_N/fused_pose vs .../goal_pose'
        )

    def _maybe_explain(self, idx: int, lines: list[str]) -> None:
        if not _as_bool(self.get_parameter('explain_motion').value):
            return
        period_s = float(self.get_parameter('explain_motion_period_s').value)
        period_ns = int(max(period_s, 0.05) * 1e9)
        now = self.get_clock().now()
        if now.nanoseconds - self._last_explain_ns[idx] < period_ns:
            return
        self._last_explain_ns[idx] = now.nanoseconds
        rid = idx + 1
        for line in lines:
            self.get_logger().info(f'[move_to_goal robot{rid}] {line}')

    def _publish_debug(self, idx: int, fused_pose: Pose, goal: Optional[Pose]) -> None:
        """Publish fused vs goal as PoseStamped for RViz / topic tools."""
        if not _as_bool(self.get_parameter('publish_debug_poses').value):
            return
        fid_default = str(self.get_parameter('debug_frame_id').value)
        stamp = self.get_clock().now().to_msg()
        last = self._last_fused_stamped[idx]
        fid = fid_default
        if last is not None and last.header.frame_id:
            fid = last.header.frame_id

        sf = PoseStamped()
        sf.header.stamp = stamp
        sf.header.frame_id = fid
        sf.pose = fused_pose
        self._debug_fused_pubs[idx].publish(sf)

        if goal is not None:
            sg = PoseStamped()
            sg.header.stamp = stamp
            sg.header.frame_id = fid
            sg.pose = goal
            self._debug_goal_pubs[idx].publish(sg)

    # ---------- Callbacks ----------

    def pose_callback(self, msg: PoseStamped, idx: int):
        """Update current fused pose for robot idx."""
        self.current_poses[idx] = msg.pose
        self._last_fused_stamped[idx] = msg
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
        position_only = _as_bool(self.get_parameter('position_only_goals').value)
        stale_s = float(self.get_parameter('pose_stale_timeout_s').value)

        now = self.get_clock().now()

        for idx in range(self.num_agents):
            pose = self.current_poses[idx]
            goal = self.goal_poses[idx]

            if pose is not None:
                self._publish_debug(idx, pose, goal)

            if pose is None:
                self.cmd_vel_pubs[idx].publish(Twist())
                self._maybe_explain(
                    idx,
                    [
                        'cmd = ZERO cmd_vel (Twist)',
                        'why: no fused pose yet (never got /fused_pose_%d)' % (idx + 1),
                    ],
                )
                continue

            if goal is None:
                self.cmd_vel_pubs[idx].publish(Twist())
                self._maybe_explain(
                    idx,
                    [
                        'cmd = ZERO cmd_vel (Twist)',
                        'why: no goal yet (never got /goal_pose_%d)' % (idx + 1),
                    ],
                )
                continue

            last_time = self.last_pose_times[idx]
            if last_time is not None:
                dt = (now - last_time).nanoseconds / 1e9
                if dt > stale_s:
                    self.cmd_vel_pubs[idx].publish(Twist())
                    self._maybe_explain(
                        idx,
                        [
                            'cmd = ZERO cmd_vel (Twist)',
                            f'why: fused pose is stale (last update {dt:.2f}s ago, '
                            f'limit {stale_s:.1f}s)',
                        ],
                    )
                    continue

            cur_x = pose.position.x
            cur_y = pose.position.y
            cur_yaw = yaw_from_quat(pose.orientation)

            goal_x = goal.position.x
            goal_y = goal.position.y
            goal_yaw_msg = yaw_from_quat(goal.orientation)
            if position_only:
                goal_yaw = cur_yaw
            else:
                goal_yaw = goal_yaw_msg

            err_x = goal_x - cur_x
            err_y = goal_y - cur_y

            err_yaw = wrap_angle(goal_yaw - cur_yaw)
            pos_err = math.hypot(err_x, err_y)

            if position_only:
                at_goal = pos_err < pos_tol
            else:
                at_goal = pos_err < pos_tol and abs(err_yaw) < yaw_tol

            stop_pos_r = float(self.get_parameter('stop_if_position_within_m').value)
            if stop_pos_r > 0.0 and pos_err < stop_pos_r:
                at_goal = True

            if at_goal:
                self.cmd_vel_pubs[idx].publish(Twist())
                yaw_note = (
                    f'yaw ignored (position_only); msg goal yaw°={math.degrees(goal_yaw_msg):.2f}'
                    if position_only
                    else (
                        f'yaw_err={math.degrees(err_yaw):.2f}° (tol {math.degrees(yaw_tol):.2f}°)'
                    )
                )
                self._maybe_explain(
                    idx,
                    [
                        'cmd = ZERO cmd_vel (Twist)',
                        'why: close enough to goal (world frame)',
                        f'  pos_err={pos_err:.4f}m (tol {pos_tol}), {yaw_note}',
                        f'  fused (x,y,yaw°)=({cur_x:.4f},{cur_y:.4f},{math.degrees(cur_yaw):.2f})',
                        f'  goal  (x,y)=( {goal_x:.4f},{goal_y:.4f})',
                    ],
                )
                continue

            rot = np.array([
                [math.cos(cur_yaw), math.sin(cur_yaw)],
                [-math.sin(cur_yaw), math.cos(cur_yaw)]
            ])
            err_x_robot, err_y_robot = rot @ np.array([err_x, err_y])

            vx_unclamped = kp_x * err_x_robot
            vy_unclamped = kp_y * err_y_robot
            wz_unclamped = kp_yaw * err_yaw

            # Scale (vx, vy) together so direction matches P output. Per-axis min/max
            # clipping alone bends large vectors toward box corners (e.g. (-1.67, 0.85)
            # became (-0.25,-0.25)), which drives the wrong way and looks like "always
            # backwards" toward walls.
            lin_scale = 1.0
            if abs(vx_unclamped) > 1e-9:
                lin_scale = min(lin_scale, max_linear_x / abs(vx_unclamped))
            if abs(vy_unclamped) > 1e-9:
                lin_scale = min(lin_scale, max_linear_y / abs(vy_unclamped))
            lin_scale = min(lin_scale, 1.0)
            vel_x_robot = vx_unclamped * lin_scale
            vel_y_robot = vy_unclamped * lin_scale

            vel_yaw = max(-max_angular_z, min(max_angular_z, wz_unclamped))

            twist = Twist()
            twist.linear.x = float(vel_x_robot)
            twist.linear.y = float(-vel_y_robot)
            twist.angular.z = float(-vel_yaw)

            self.cmd_vel_pubs[idx].publish(twist)

            clamp_bits = []
            if lin_scale < 1.0 - 1e-6:
                clamp_bits.append(
                    f'holonomic linear scaled by {lin_scale:.3f} '
                    f'(limits ±{max_linear_x}, ±{max_linear_y}; '
                    f'raw vx={vx_unclamped:.3f} vy={vy_unclamped:.3f})'
                )
            if abs(wz_unclamped - vel_yaw) > 1e-6:
                clamp_bits.append(f'wz clamped to ±{max_angular_z} (raw {wz_unclamped:.3f})')
            clamp_line = '; '.join(clamp_bits) if clamp_bits else 'no saturation on this tick'

            why_move = (
                f'why: pos_err={pos_err:.4f}m > {pos_tol} (position_only={position_only})'
                if position_only
                else (
                    f'why: pos_err={pos_err:.4f}m > {pos_tol} or '
                    f'|yaw_err|={math.degrees(abs(err_yaw)):.2f}° > {math.degrees(yaw_tol):.2f}°'
                )
            )
            self._maybe_explain(
                idx,
                [
                    'cmd = non-zero cmd_vel (P-control toward /goal_pose in robot frame)',
                    why_move,
                    f'  world err goal-fused: dx={err_x:.4f} dy={err_y:.4f} (m)',
                    f'  body err (before P): ex_b={err_x_robot:.4f} ey_b={err_y_robot:.4f} yaw_err={math.degrees(err_yaw):.2f}°',
                    f'  P raw: vx={vx_unclamped:.4f} vy={vy_unclamped:.4f} wz={wz_unclamped:.4f} rad/s',
                    f'  published Twist: linear.x={twist.linear.x:.4f} linear.y={twist.linear.y:.4f} angular.z={twist.angular.z:.4f} (signs per RoboMaster convention)',
                    f'  {clamp_line}',
                ],
            )

    # ---------- Utility ----------

    def publish_stop_all(self):
        """Publish zero velocities to all robots."""
        if not rclpy.ok():
            return
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
        try:
            node.publish_stop_all()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()