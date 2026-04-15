#!/usr/bin/env python3
"""
Swap workflow: read two fused poses, plan collision-free grid paths with CBS,
then publish synchronized pose goals (position + yaw) for each robot.

Assumes /fused_pose_1, /fused_pose_2 (PoseStamped, frame world) and
move_to_goal subscribed to /goal_pose_1, /goal_pose_2 (Pose).
"""
from __future__ import annotations

import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import String, UInt32

from .cbs import cbs
from . import cbs as cbs_module


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class Phase(Enum):
    WAIT_POSES = auto()
    EXECUTE = auto()
    DONE = auto()


class SwapPlanner(Node):
    """Plan a two-robot position swap using CBS on a grid, then stream pose goals."""

    def __init__(self):
        super().__init__('swap_planner')

        self.declare_parameter('cell_size_m', 0.25)
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('step_period_s', 3.5)
        self.declare_parameter('plan_retry_s', 0.5)
        self.declare_parameter('publish_debug_paths', True)
        self.declare_parameter('debug_path_frame_id', 'world')

        self._cell = float(self.get_parameter('cell_size_m').value)
        self._ox = float(self.get_parameter('origin_x').value)
        self._oy = float(self.get_parameter('origin_y').value)

        self._agents = (1, 2)
        self._fused_last: dict[int, PoseStamped | None] = {1: None, 2: None}
        self._phase = Phase.WAIT_POSES
        self._planned = False
        self._padded: dict[int, list[tuple[int, int]]] = {}
        self._goal_yaws: dict[int, float] = {}
        self._ti = 0
        self._t_max = 0

        self._goal_pub = {
            1: self.create_publisher(Pose, '/goal_pose_1', 10),
            2: self.create_publisher(Pose, '/goal_pose_2', 10),
        }
        self._path_pub = {
            1: self.create_publisher(Path, '/swap_planner/debug/robot_1/planned_path', 10),
            2: self.create_publisher(Path, '/swap_planner/debug/robot_2/planned_path', 10),
        }
        self._idx_pub = self.create_publisher(
            UInt32, '/swap_planner/debug/sync_plan_index', 10
        )

        _qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._final_fused_pub = {
            1: self.create_publisher(
                PoseStamped,
                '/swap_planner/debug/final_fused_pose_1',
                _qos_latched,
            ),
            2: self.create_publisher(
                PoseStamped,
                '/swap_planner/debug/final_fused_pose_2',
                _qos_latched,
            ),
        }
        self._summary_pub = self.create_publisher(
            String,
            '/swap_planner/debug/mission_summary',
            _qos_latched,
        )

        self.create_subscription(PoseStamped, '/fused_pose_1', self._pose1_cb, 10)
        self.create_subscription(PoseStamped, '/fused_pose_2', self._pose2_cb, 10)

        self._exec_timer = None

        retry = float(self.get_parameter('plan_retry_s').value)
        self._wait_timer = self.create_timer(retry, self._on_wait_tick)

        self.get_logger().info(
            'swap_planner: waiting for /fused_pose_1 and /fused_pose_2; '
            f'grid cell={self._cell} m, CBS bounds ±{cbs_module.grid_scale_factor}'
        )
        self.get_logger().info(
            '  Debug: /swap_planner/debug/robot_N/planned_path (nav_msgs/Path), '
            '/swap_planner/debug/sync_plan_index (uint32)'
        )
        self.get_logger().info(
            '  On plan complete: transient_local /swap_planner/debug/final_fused_pose_N, '
            'mission_summary (String)'
        )

    def _pose1_cb(self, msg: PoseStamped):
        self._fused_last[1] = msg

    def _pose2_cb(self, msg: PoseStamped):
        self._fused_last[2] = msg

    def _publish_mission_done_snapshot(self) -> None:
        """Latched-style topics so late RViz / echo still see last fused poses."""
        lines = [
            'swap_planner: mission done (final goals held). Fused snapshot:',
        ]
        for a in self._agents:
            src = self._fused_last[a]
            if src is None:
                lines.append(f'  robot{a}: (no /fused_pose yet)')
                continue
            out = PoseStamped()
            out.header = src.header
            out.header.stamp = self.get_clock().now().to_msg()
            out.pose = src.pose
            self._final_fused_pub[a].publish(out)
            x = src.pose.position.x
            y = src.pose.position.y
            ydeg = math.degrees(yaw_from_quat(src.pose.orientation))
            lines.append(f'  robot{a}: x={x:.4f} y={y:.4f} yaw_deg={ydeg:.2f}')
        summary = '\n'.join(lines)
        self._summary_pub.publish(String(data=summary))
        for line in lines:
            self.get_logger().info(line)

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(round((x - self._ox) / self._cell))
        gy = int(round((y - self._oy) / self._cell))
        lim = cbs_module.grid_scale_factor
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = self._ox + float(gx) * self._cell
        wy = self._oy + float(gy) * self._cell
        return wx, wy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        lim = cbs_module.grid_scale_factor
        return -lim <= gx <= lim and -lim <= gy <= lim

    def _yaw_along_path(self, path: list[tuple[int, int]], k: int, final_yaw: float) -> float:
        cell = path[k]
        m = k + 1
        while m < len(path) and path[m] == cell:
            m += 1
        if m < len(path):
            wx, wy = self._grid_to_world(*cell)
            wxn, wyn = self._grid_to_world(*path[m])
            return math.atan2(wyn - wy, wxn - wx)
        return final_yaw

    def _make_pose(self, gx: int, gy: int, yaw: float) -> Pose:
        wx, wy = self._grid_to_world(gx, gy)
        p = Pose()
        p.position.x = float(wx)
        p.position.y = float(wy)
        p.position.z = 0.0
        p.orientation = quat_from_yaw(yaw)
        return p

    def _debug_path_for_agent(self, agent: int) -> Path:
        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = str(self.get_parameter('debug_path_frame_id').value)
        grid_path = self._padded[agent]
        for k in range(len(grid_path)):
            gx, gy = grid_path[k]
            yaw = self._yaw_along_path(grid_path, k, self._goal_yaws[agent])
            ps = PoseStamped()
            ps.header = out.header
            ps.pose = self._make_pose(gx, gy, yaw)
            out.poses.append(ps)
        return out

    def _publish_debug_paths(self, time_index: int) -> None:
        if not _as_bool(self.get_parameter('publish_debug_paths').value):
            return
        for a in self._agents:
            self._path_pub[a].publish(self._debug_path_for_agent(a))
        msg = UInt32()
        msg.data = int(time_index)
        self._idx_pub.publish(msg)

    def _on_wait_tick(self):
        if self._planned:
            return
        if self._fused_last[1] is None or self._fused_last[2] is None:
            return

        p1 = self._fused_last[1].pose
        p2 = self._fused_last[2].pose

        x1, y1 = p1.position.x, p1.position.y
        x2, y2 = p2.position.x, p2.position.y
        s1 = self._world_to_grid(x1, y1)
        s2 = self._world_to_grid(x2, y2)

        if not self._in_bounds(*s1) or not self._in_bounds(*s2):
            self.get_logger().warn(
                f'Robot grid starts {s1}, {s2} outside CBS ±{cbs_module.grid_scale_factor}. '
                'Adjust origin_x/y or cell_size_m.'
            )
            return

        if s1 == s2:
            self.get_logger().warn('Both robots map to the same cell; move them apart slightly.')
            return

        yaw1 = yaw_from_quat(p1.orientation)
        yaw2 = yaw_from_quat(p2.orientation)

        starts = {1: s1, 2: s2}
        goals = {1: s2, 2: s1}

        solution = cbs(list(self._agents), starts, goals)
        if not solution:
            self.get_logger().error('CBS found no swap plan for this grid configuration.')
            return

        max_len = max(len(solution[a]) for a in self._agents)
        padded: dict[int, list[tuple[int, int]]] = {}
        for a in self._agents:
            path = list(solution[a])
            last = path[-1]
            if len(path) < max_len:
                path = path + [last] * (max_len - len(path))
            padded[a] = path

        self._padded = padded
        self._goal_yaws = {1: yaw2, 2: yaw1}
        self._t_max = max_len
        self._ti = 0
        self._planned = True
        self._wait_timer.cancel()

        self.get_logger().info(f'CBS swap plan: agent1 path {solution[1]}')
        self.get_logger().info(f'CBS swap plan: agent2 path {solution[2]}')
        self.get_logger().info(
            f'Padded length {max_len}, publishing every '
            f'{self.get_parameter("step_period_s").value} s (synced index for both).'
        )

        period = float(self.get_parameter('step_period_s').value)
        if self._exec_timer is not None:
            self._exec_timer.cancel()
        self._ti = min(1, self._t_max - 1)
        self._publish_step(self._ti)
        self._exec_timer = self.create_timer(period, self._on_exec_tick)
        self._phase = Phase.EXECUTE

    def _publish_step(self, ti: int):
        ti = max(0, min(ti, self._t_max - 1))
        for a in self._agents:
            path = self._padded[a]
            gx, gy = path[ti]
            yaw = self._yaw_along_path(path, ti, self._goal_yaws[a])
            pose = self._make_pose(gx, gy, yaw)
            self._goal_pub[a].publish(pose)
        self._publish_debug_paths(ti)

    def _on_exec_tick(self):
        if not self._planned:
            return
        if self._ti >= self._t_max - 1:
            self._publish_step(self._t_max - 1)
            if self._phase != Phase.DONE:
                self.get_logger().info(
                    'swap_planner: plan complete; final /goal_pose_* published. '
                    'Stopping exec timer (goals unchanged). See mission_summary + final_fused_pose_*.'
                )
                self._phase = Phase.DONE
                self._publish_mission_done_snapshot()
                if self._exec_timer is not None:
                    self._exec_timer.cancel()
                    self._exec_timer = None
            return
        self._ti += 1
        self._publish_step(self._ti)


def main(args=None):
    rclpy.init(args=args)
    node = SwapPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
