#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion

# ✅ Import your existing CBS implementation
# Make sure this matches the actual file/module name where your cbs() lives.
from .cbs import cbs

scale_factor = 2

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class CBSPathTester(Node):
    """
    - Calls your CBS algorithm once for 2 agents with hardcoded starts/goals.
    - Then, every second, publishes the next coordinate on each agent's path
      to /goal_pose_1 and /goal_pose_2 (yaw = 0).
    """

    def __init__(self):
        super().__init__('cbs_path_tester')

        # Hardcoded 2 agents and starts/goals (exactly like your example)
        self.agents = [1, 2]
        self.starts = {
            1: (0.5, 0),
            2: (0.5, 1),
        }
        self.goals = {
            1: (0.5, 1),
            2: (0.5, 0),
        }

        self.starts = {
            k: (v[0] * scale_factor, v[1] * scale_factor)
            for k, v in self.starts.items()
        }
        self.goals = {
            k: (v[0] * scale_factor, v[1] * scale_factor)
            for k, v in self.goals.items()
        }

        self.get_logger().info("Running CBS for 2 agents...")
        self.get_logger().info(f"  Starts: {self.starts}")
        self.get_logger().info(f"  Goals:  {self.goals}")

        # Run CBS once
        solution = cbs(self.agents, self.starts, self.goals)

        if not solution:
            self.get_logger().error("No solution found by CBS.")
            self.paths = {1: [], 2: []}
        else:
            # solution[agent] is a list of (x, y)
            self.paths = {
                agent: list(path)   # make a copy we can pop from
                for agent, path in solution.items()
            }
            for agent, path in self.paths.items():
                self.get_logger().info(f"Agent {agent} path: {path}")

        # Keep track of remaining waypoints and the last goal for each agent
        self.remaining = {
            agent: list(self.paths.get(agent, []))
            for agent in self.agents
        }
        self.last_goal = {agent: None for agent in self.agents}

        # Publishers for /goal_pose_1 and /goal_pose_2
        self.goal_pubs = {
            1: self.create_publisher(Pose, '/goal_pose_1', 10),
            2: self.create_publisher(Pose, '/goal_pose_2', 10),
        }

        # Timer: every second, pop & publish next coordinate
        self.timer = self.create_timer(2.0, self.timer_callback)

        self.get_logger().info("CBSPathTester started. Publishing waypoints once per second.")

    def timer_callback(self):
        """Pop one coordinate from each agent's path and publish as a Pose."""
        yaw = 0.0  # keep yaw = 0 for now

        for agent in self.agents:
            pub = self.goal_pubs[agent]

            # If we still have waypoints left, pop the next one
            if self.remaining[agent]:
                x, y = self.remaining[agent].pop(0)
                self.last_goal[agent] = (x, y)
                self.get_logger().info(
                    f"[Agent {agent}] New waypoint: ({x}, {y}), "
                    f"{len(self.remaining[agent])} remaining."
                )
            else:
                # No more waypoints: keep re-publishing the last goal if we have one
                if self.last_goal[agent] is None:
                    # No path / nothing to send
                    continue
                x, y = self.last_goal[agent]
                # You can comment this log out if it's too spammy
                self.get_logger().info(
                    f"[Agent {agent}] Re-publishing final goal: ({x}, {y})"
                )

            # Build and publish Pose
            pose = Pose()
            pose.position.x = float(x) / scale_factor
            pose.position.y = float(y) / scale_factor
            pose.position.z = 0.0
            pose.orientation = quat_from_yaw(yaw)
            pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = CBSPathTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()