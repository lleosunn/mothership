#!/usr/bin/env python3
"""
Formation planner: auto-discover active agents, then command them into a
straight line ordered by agent ID (lowest → highest).

  axis='x'  → horizontal line (same y, spaced along x)
  axis='y'  → vertical line   (same x, spaced along y)

The formation center is the centroid of the detected agents' current
positions.  Spacing between neighbours equals the cell_size parameter.

DISCOVER → PLAN → HOLD

Relies on move_to_goal for the actual motion control.
"""
from __future__ import annotations

import math
import time as _time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Quaternion


CANDIDATE_AGENT_IDS = [1, 2, 3, 4, 7]


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class FormationPlanner(Node):
    def __init__(self):
        super().__init__('formation_planner')

        self.declare_parameter('axis', 'horizontal')
        self.declare_parameter('spacing_m', 0.48)
        self.declare_parameter('discovery_timeout_s', 3.0)
        self.declare_parameter('min_agents', 2)
        self.declare_parameter('publish_rate', 10.0)

        self._axis = str(self.get_parameter('axis').value)
        self._spacing = float(self.get_parameter('spacing_m').value)
        self._discovery_timeout = float(self.get_parameter('discovery_timeout_s').value)
        self._min_agents = int(self.get_parameter('min_agents').value)

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

        self._agents: list[int] = []
        self._fused: dict[int, PoseStamped | None] = {}
        self._goal_pubs: dict[int, rclpy.publisher.Publisher] = {}
        self._goals: dict[int, Pose] = {}
        self._hold_timer = None

        self.get_logger().info(
            f'formation_planner: DISCOVER phase — candidates {self._candidates}, '
            f'axis={self._axis}, spacing={self._spacing}m'
        )

    def _pose_cb(self, msg: PoseStamped, agent: int):
        self._candidate_fused[agent] = msg
        if agent in self._fused:
            self._fused[agent] = msg

    # ---- Discovery ----

    def _check_discovery(self):
        elapsed = _time.monotonic() - self._discovery_start
        detected = sorted(a for a in self._candidates if self._candidate_fused[a] is not None)

        self.get_logger().info(
            f'Discovery {elapsed:.1f}s/{self._discovery_timeout}s: '
            f'detected {detected}'
        )

        if elapsed < self._discovery_timeout:
            if len(detected) == len(self._candidates):
                self.get_logger().info(f'All {len(detected)} candidates detected early: {detected}')
                self._finalize_discovery(detected)
            return

        if len(detected) < self._min_agents:
            self.get_logger().error(
                f'Discovery timeout: only {len(detected)} agents {detected}, '
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
        self._goal_pubs = {
            a: self.create_publisher(Pose, f'/goal_pose_{a}', 10)
            for a in agents
        }

        self.get_logger().info(f'Discovery complete: {len(agents)} agents {agents}')
        self._compute_formation()

    # ---- Formation computation ----

    def _compute_formation(self):
        n = len(self._agents)

        cx = sum(self._fused[a].pose.position.x for a in self._agents) / n
        cy = sum(self._fused[a].pose.position.y for a in self._agents) / n

        total_span = (n - 1) * self._spacing
        offsets = [-total_span / 2.0 + i * self._spacing for i in range(n)]

        self._goals = {}
        for i, agent_id in enumerate(self._agents):
            pose = Pose()
            if self._axis in ('x', 'horizontal'):
                pose.position.x = cx + offsets[i]
                pose.position.y = cy
            else:
                pose.position.x = cx
                pose.position.y = cy + offsets[i]
            pose.position.z = 0.0
            pose.orientation = quat_from_yaw(0.0)
            self._goals[agent_id] = pose

        self._phase = 'HOLD'
        rate = float(self.get_parameter('publish_rate').value)
        self._hold_timer = self.create_timer(1.0 / rate, self._publish_goals)

        self.get_logger().info(
            f'Formation ({self._axis}-axis line) centroid=({cx:.3f}, {cy:.3f}):'
        )
        for a in self._agents:
            g = self._goals[a]
            self.get_logger().info(
                f'  Agent {a} → x={g.position.x:.3f}, y={g.position.y:.3f}'
            )

    def _publish_goals(self):
        for a in self._agents:
            self._goal_pubs[a].publish(self._goals[a])


def main(args=None):
    rclpy.init(args=args)
    node = FormationPlanner()
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
