#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose


def quat_from_yaw(yaw: float):
    """Create a geometry_msgs/Quaternion from a yaw angle."""
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


def pose_from_xytheta(x: float, y: float, theta: float) -> Pose:
    """Create a Pose from (x, y, theta) with z=0 and yaw=theta."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0
    pose.orientation = quat_from_yaw(theta)
    return pose


class TwoAgentGoalTester(Node):
    def __init__(self):
        super().__init__('two_agent_goal_tester')

        # Fixed 2 agents for this tester
        self.num_agents = 2

        # Publishers for goal poses
        self.goal_pubs = [
            self.create_publisher(Pose, '/goal_pose_1', 10),
            self.create_publisher(Pose, '/goal_pose_2', 10),
        ]

        # Define the looping sequence of goals:
        # Each entry is: [(x1, y1, theta1), (x2, y2, theta2)]
        self.goal_sequence = [
            # t = 0s → start
            [(0.0, 0.0, 0.0), (0.4, 0.4, 0.0)],
            # t = 1s
            [(0.4, 0.0, 0.0), (0.0, 0.4, 0.0)],
            # t = 2s
            [(0.4, 0.4, 0.0), (0.0, 0.0, 0.0)],

            [(0.0, 0.4, 0.0), (0.4, 0.0, 0.0)],
        ]

        self.current_index = 0  # which element of goal_sequence we're on

        # Timer to advance to the next set of goals every 1 second
        self.phase_timer = self.create_timer(1.0, self.advance_goals)

        # Timer to continuously publish current goals at 10 Hz
        self.publish_timer = self.create_timer(0.1, self.publish_current_goals)

        self.get_logger().info("🎯 TwoAgentGoalTester started")
        self.log_current_goals()

    # ---------- Timers ----------

    def advance_goals(self):
        """Advance to the next pair of goals in the sequence (1 Hz)."""
        self.current_index = (self.current_index + 1) % len(self.goal_sequence)
        self.log_current_goals()

    def publish_current_goals(self):
        """Publish the current goals for both agents (10 Hz)."""
        goals = self.goal_sequence[self.current_index]

        # Agent 1
        g1 = goals[0]
        pose1 = pose_from_xytheta(*g1)
        self.goal_pubs[0].publish(pose1)

        # Agent 2
        g2 = goals[1]
        pose2 = pose_from_xytheta(*g2)
        self.goal_pubs[1].publish(pose2)

    # ---------- Helper ----------

    def log_current_goals(self):
        """Log the current goal pair for debugging."""
        goals = self.goal_sequence[self.current_index]
        (x1, y1, th1) = goals[0]
        (x2, y2, th2) = goals[1]
        self.get_logger().info(
            f"New goals [index {self.current_index}]: "
            f"Agent 1 -> ({x1:.2f}, {y1:.2f}, {math.degrees(th1):.1f}°), "
            f"Agent 2 -> ({x2:.2f}, {y2:.2f}, {math.degrees(th2):.1f}°)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TwoAgentGoalTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()