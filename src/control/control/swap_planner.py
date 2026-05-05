#!/usr/bin/env python3
"""
Swap planner: auto-discover which agents are publishing fused poses,
build a cyclic swap among them, plan with CBS, then execute.

DISCOVER → WAIT → ALIGN → EXECUTE → DONE

Relies on move_to_goal for the actual motion control.
"""
from __future__ import annotations

import math
import time as _time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Quaternion

from .cbs import cbs, min_separation_for_cell_size
from . import cbs as cbs_module


CANDIDATE_AGENT_IDS = [1, 2, 3, 4, 7]


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


def build_cyclic_goal_map(agents: list[int]) -> dict[int, int]:
    """Build a cyclic swap: each agent goes to the next agent's position."""
    n = len(agents)
    return {agents[i]: agents[(i + 1) % n] for i in range(n)}


class SwapPlanner(Node):
    def __init__(self):
        super().__init__('swap_planner')

        self.declare_parameter('cell_size_m', 0.48)
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('step_period_s', 1.0)
        self.declare_parameter('plan_retry_s', 0.5)
        self.declare_parameter('align_tolerance_m', 0.10)
        self.declare_parameter('align_yaw_tolerance_rad', 0.20)
        self.declare_parameter('align_check_rate', 10.0)
        self.declare_parameter('discovery_timeout_s', 3.0)
        self.declare_parameter('min_agents', 2)

        self._cell = float(self.get_parameter('cell_size_m').value)
        self._ox = float(self.get_parameter('origin_x').value)
        self._oy = float(self.get_parameter('origin_y').value)
        self._align_tol = float(self.get_parameter('align_tolerance_m').value)
        self._align_yaw_tol = float(self.get_parameter('align_yaw_tolerance_rad').value)
        self._min_sep = min_separation_for_cell_size(self._cell)
        self._discovery_timeout = float(self.get_parameter('discovery_timeout_s').value)
        self._min_agents = int(self.get_parameter('min_agents').value)

        # --- Discovery phase: subscribe to all candidates, see who responds ---
        self._candidates = list(CANDIDATE_AGENT_IDS)
        self._candidate_fused: dict[int, PoseStamped | None] = {a: None for a in self._candidates}
        self._discovery_start = _time.monotonic()
        self._phase = 'DISCOVER'

        for a in self._candidates:
            self.create_subscription(
                PoseStamped, f'/fused_pose_{a}',
                lambda msg, agent=a: self._pose_cb(msg, agent), 10
            )

        self._discover_timer = self.create_timer(0.25, self._check_discovery)

        # Populated after discovery
        self._agents: list[int] = []
        self._fused: dict[int, PoseStamped | None] = {}
        self._goal_map: dict[int, int] = {}
        self._goal_pubs: dict[int, rclpy.publisher.Publisher] = {}
        self._align_targets: dict[int, tuple[float, float]] = {}
        self._remaining: dict[int, list[tuple[int, int]]] = {}
        self._last_goal: dict[int, Pose | None] = {}

        self._wait_timer = None
        self._align_timer = None
        self._step_timer = None

        self.get_logger().info(
            f'swap_planner: DISCOVER phase — listening for candidates {self._candidates}, '
            f'timeout={self._discovery_timeout}s, min_agents={self._min_agents}'
        )

    # ---- Discovery ----

    def _pose_cb(self, msg: PoseStamped, agent: int):
        self._candidate_fused[agent] = msg
        if agent in self._fused:
            self._fused[agent] = msg

    def _check_discovery(self):
        elapsed = _time.monotonic() - self._discovery_start
        detected = sorted(a for a in self._candidates if self._candidate_fused[a] is not None)

        if elapsed < self._discovery_timeout:
            if len(detected) == len(self._candidates):
                self.get_logger().info(
                    f'All {len(detected)} candidates detected early: {detected}'
                )
                self._finalize_discovery(detected)
            return

        if len(detected) < self._min_agents:
            self.get_logger().error(
                f'Discovery timeout: only {len(detected)} agents detected {detected}, '
                f'need at least {self._min_agents}. Retrying...'
            )
            self._discovery_start = _time.monotonic()
            for a in self._candidates:
                self._candidate_fused[a] = None
            return

        self._finalize_discovery(detected)

    def _finalize_discovery(self, agents: list[int]):
        self._discover_timer.cancel()
        self._agents = agents
        self._fused = {a: self._candidate_fused[a] for a in agents}
        self._goal_map = build_cyclic_goal_map(agents)
        self._remaining = {a: [] for a in agents}
        self._last_goal = {a: None for a in agents}

        self._goal_pubs = {
            a: self.create_publisher(Pose, f'/goal_pose_{a}', 10)
            for a in agents
        }

        self._phase = 'WAIT'
        retry = float(self.get_parameter('plan_retry_s').value)
        self._wait_timer = self.create_timer(retry, self._try_plan)

        self.get_logger().info(
            f'Discovery complete: {len(agents)} agents {agents}; '
            f'grid cell={self._cell} m, min_separation={self._min_sep:.2f} cells, '
            f'CBS bounds ±{cbs_module.grid_scale_factor}'
        )
        self.get_logger().info(f'  Cyclic swap: {self._goal_map}')

    # ---- Planning & execution (unchanged logic, uses self._agents) ----

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(round((x - self._ox) / self._cell))
        gy = int(round((y - self._oy) / self._cell))
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = self._ox + float(gx) * self._cell
        wy = self._oy + float(gy) * self._cell
        return wx, wy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        lim = cbs_module.grid_scale_factor
        return -lim <= gx <= lim and -lim <= gy <= lim

    def _try_plan(self):
        if self._phase != 'WAIT':
            return
        if any(self._fused[a] is None for a in self._agents):
            return

        grid_starts = {}
        for a in self._agents:
            p = self._fused[a].pose
            grid_starts[a] = self._world_to_grid(p.position.x, p.position.y)

        for a in self._agents:
            if not self._in_bounds(*grid_starts[a]):
                self.get_logger().warn(
                    f'Agent {a} grid start {grid_starts[a]} outside CBS '
                    f'±{cbs_module.grid_scale_factor}. Adjust origin/cell.'
                )
                return

        cells = list(grid_starts.values())
        if len(set(cells)) != len(cells):
            self.get_logger().warn('Two or more robots map to the same cell; move them apart.')
            return

        starts = {a: grid_starts[a] for a in self._agents}
        goals = {a: grid_starts[self._goal_map[a]] for a in self._agents}

        self.get_logger().info(f'Planning CBS cyclic swap ({len(self._agents)} agents):')
        for a in self._agents:
            self.get_logger().info(f'  Agent {a}: {starts[a]} → {goals[a]} (agent {self._goal_map[a]}\'s spot)')

        solution = cbs(list(self._agents), starts, goals, min_separation=self._min_sep)
        if not solution:
            self.get_logger().error('CBS found no swap solution.')
            return

        max_len = max(len(solution[a]) for a in self._agents)
        for a in self._agents:
            path = list(solution[a])
            if len(path) < max_len:
                path = path + [path[-1]] * (max_len - len(path))
            self._remaining[a] = path

        self._wait_timer.cancel()

        for a in self._agents:
            self.get_logger().info(f'  Agent {a} path: {self._remaining[a]}')

        self._align_targets = {}
        for a in self._agents:
            gx, gy = self._remaining[a][0]
            wx, wy = self._grid_to_world(gx, gy)
            self._align_targets[a] = (wx, wy)
            pose = Pose()
            pose.position.x = float(wx)
            pose.position.y = float(wy)
            pose.position.z = 0.0
            pose.orientation = quat_from_yaw(0.0)
            self._last_goal[a] = pose
            self._goal_pubs[a].publish(pose)

        self._phase = 'ALIGN'
        self._align_start = _time.monotonic()
        self._align_max_s = 10.0
        self._align_log_interval = 1.0
        self._align_last_log = 0.0
        align_rate = float(self.get_parameter('align_check_rate').value)
        self._align_timer = self.create_timer(1.0 / align_rate, self._check_alignment)
        self.get_logger().info(
            f'ALIGN phase (max {self._align_max_s:.0f}s): '
            f'robots moving to grid cell centers with yaw=0. '
            f'Targets: {self._align_targets}'
        )

    def _finish_align(self):
        self._align_timer.cancel()
        self._align_timer = None
        self._phase = 'EXECUTE'
        for a in self._agents:
            self._remaining[a].pop(0)
        period = float(self.get_parameter('step_period_s').value)
        self._step_timer = self.create_timer(period, self._step_callback)

    def _check_alignment(self):
        elapsed = _time.monotonic() - self._align_start
        all_aligned = True
        status_parts = []

        for a in self._agents:
            fused = self._fused[a]
            if fused is None:
                all_aligned = False
                status_parts.append(f'  Agent {a}: no pose')
                continue

            tx, ty = self._align_targets[a]
            cx = fused.pose.position.x
            cy = fused.pose.position.y
            cyaw = yaw_from_quat(fused.pose.orientation)

            pos_err = math.hypot(tx - cx, ty - cy)
            yaw_err = abs(cyaw)
            ok = pos_err <= self._align_tol and yaw_err <= self._align_yaw_tol

            if not ok:
                all_aligned = False

            status_parts.append(
                f'  Agent {a}: pos_err={pos_err:.3f}m yaw_err={math.degrees(yaw_err):.1f}° '
                f'({"OK" if ok else "moving"})'
            )

            self._goal_pubs[a].publish(self._last_goal[a])

        if elapsed - self._align_last_log >= self._align_log_interval:
            self._align_last_log = elapsed
            self.get_logger().info(
                f'ALIGN [{elapsed:.1f}s / {self._align_max_s:.0f}s]:'
            )
            for part in status_parts:
                self.get_logger().info(part)

        if all_aligned:
            self.get_logger().info(
                f'ALIGN complete after {elapsed:.1f}s. Starting CBS path execution.'
            )
            self._finish_align()
            return

        if elapsed >= self._align_max_s:
            self.get_logger().warn(
                f'ALIGN timeout ({self._align_max_s:.0f}s). '
                f'Proceeding with execution anyway.'
            )
            for part in status_parts:
                self.get_logger().warn(part)
            self._finish_align()

    def _step_callback(self):
        yaw = 0.0

        for a in self._agents:
            if self._remaining[a]:
                gx, gy = self._remaining[a].pop(0)
                wx, wy = self._grid_to_world(gx, gy)
                pose = Pose()
                pose.position.x = float(wx)
                pose.position.y = float(wy)
                pose.position.z = 0.0
                pose.orientation = quat_from_yaw(yaw)
                self._last_goal[a] = pose
            elif self._last_goal[a] is None:
                continue

            self._goal_pubs[a].publish(self._last_goal[a])

        if all(len(self._remaining[a]) == 0 for a in self._agents):
            self._phase = 'DONE'
            self.get_logger().info('Swap complete. Holding final goals.')


def main(args=None):
    rclpy.init(args=args)
    node = SwapPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
