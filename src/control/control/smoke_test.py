#!/usr/bin/env python3
"""
Smoke test: read an agent's fused pose, then command it to move to a
position offset by (dx, dy) metres from where it currently is.

Publishes a single goal and keeps re-publishing it so move_to_goal
stays engaged.  Shuts down once the robot is within tolerance.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Quaternion


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SmokeTest(Node):
    def __init__(self):
        super().__init__('smoke_test')

        self.declare_parameter('agent_id', 4)
        self.declare_parameter('dx', 0.3)
        self.declare_parameter('dy', 0.0)
        self.declare_parameter('tolerance', 0.05)
        self.declare_parameter('publish_rate', 10.0)

        self._agent = int(self.get_parameter('agent_id').value)
        self._dx = float(self.get_parameter('dx').value)
        self._dy = float(self.get_parameter('dy').value)
        self._tol = float(self.get_parameter('tolerance').value)

        self._goal: Pose | None = None
        self._current: Pose | None = None

        self._goal_pub = self.create_publisher(
            Pose, f'/goal_pose_{self._agent}', 10
        )

        self.create_subscription(
            PoseStamped, f'/fused_pose_{self._agent}',
            self._pose_cb, 10
        )

        rate = float(self.get_parameter('publish_rate').value)
        self._timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'Smoke test: agent {self._agent}, '
            f'will offset current pose by dx={self._dx}, dy={self._dy}'
        )

    def _pose_cb(self, msg: PoseStamped):
        self._current = msg.pose

        if self._goal is None:
            goal = Pose()
            goal.position.x = msg.pose.position.x + self._dx
            goal.position.y = msg.pose.position.y + self._dy
            goal.position.z = 0.0
            goal.orientation = msg.pose.orientation
            self._goal = goal
            self.get_logger().info(
                f'Start  : x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
            )
            self.get_logger().info(
                f'Goal   : x={goal.position.x:.3f}, y={goal.position.y:.3f}'
            )

    def _tick(self):
        if self._goal is None or self._current is None:
            return

        self._goal_pub.publish(self._goal)

        ex = self._goal.position.x - self._current.position.x
        ey = self._goal.position.y - self._current.position.y
        dist = math.hypot(ex, ey)

        if dist < self._tol:
            self.get_logger().info(
                f'Reached goal (error={dist:.3f} m). Smoke test PASSED.'
            )
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = SmokeTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
