#!/usr/bin/env python3
"""
ROS2 Collision Avoidance Node
============================

This node implements the VMAS collision avoidance system for ROS2 Humble.
It publishes Twist messages to control two robots:
- Robot 1: /robomaster_1/cmd_vel
- Robot 2: /robomaster_2/cmd_vel

The force vectors from the collision avoidance system are converted to
linear velocities in the Twist messages.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random
import time
import math
import torch
import numpy as np
import heapq
from collections import deque
from scipy.interpolate import splprep, splev
import typing
from typing import Callable, Dict, List

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Collision avoidance parameters
grid_scale_factor = 5
kp = 0.8
following_distance = 0.15
avoid_radius = 0.15
repulse_strength = 0.05
max_force = 0.1
spline_error = 0

# Robot start and goal positions
starts = [
    [0.0, -1.0], 
    [0.0, 1.0],
]
goals = [
    [0.0, 1.0],
    [0.0, -1.0],
]

# =============================================================================
# CBS (Conflict-Based Search) ALGORITHM
# =============================================================================

class CBSNode:
    def __init__(self, constraints, solution, cost):
        self.constraints = constraints
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def astar(agent, start, goal, constraints):
    vertex_constraints = {
        (c['loc'], c['time'])
        for c in constraints
        if c['agent'] == agent and not isinstance(c['loc'][0], tuple)
    }
    edge_constraints = {
        (c['loc'][0], c['loc'][1], c['time'])
        for c in constraints
        if c['agent'] == agent and isinstance(c['loc'][0], tuple)
    }

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_list = [(h(start), 0, start, [start])]
    best_g = {}

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        time = g

        if current == goal:
            return path

        key = (current, time)
        if key in best_g and g >= best_g[key]:
            continue
        best_g[key] = g

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            if not (-grid_scale_factor <= nx <= grid_scale_factor and -grid_scale_factor <= ny <= grid_scale_factor):
                continue
            new_time = time + 1

            if (next_pos, new_time) in vertex_constraints:
                continue
            if (current, next_pos, new_time) in edge_constraints:
                continue

            new_g = g + 1
            new_f = new_g + h(next_pos)
            heapq.heappush(open_list, (new_f, new_g, next_pos, path + [next_pos]))

    return None


def detect_conflict(paths):
    max_time = max(len(p) for p in paths.values())
    for t in range(max_time):
        pos_at_t = {}
        for a, path in paths.items():
            pos = path[min(t, len(path) - 1)]
            if pos in pos_at_t:
                return {'type': 'vertex', 'time': t, 'a1': pos_at_t[pos], 'a2': a, 'loc': pos}
            pos_at_t[pos] = a

        from itertools import combinations
        for a1, a2 in combinations(paths.keys(), 2):
            pos1_t = paths[a1][min(t, len(paths[a1]) - 1)]
            pos1_t1 = paths[a1][min(t + 1, len(paths[a1]) - 1)]
            pos2_t = paths[a2][min(t, len(paths[a2]) - 1)]
            pos2_t1 = paths[a2][min(t + 1, len(paths[a2]) - 1)]
            if pos1_t == pos2_t1 and pos2_t == pos1_t1:
                return {
                    'type': 'edge',
                    'time': t + 1,
                    'a1': a1,
                    'a2': a2,
                    'loc': (pos1_t, pos1_t1)
                }
    return None


def compute_solution(agents, constraints, starts, goals):
    solution = {}
    for agent in agents:
        path = astar(agent, starts[agent], goals[agent], constraints)
        if not path:
            return None
        solution[agent] = path
    return solution


def compute_cost(solution):
    if solution is None:
        return float("inf")
    return sum(max(len(path) - 1, 0) for path in solution.values())


def cbs(agents, starts, goals):
    root_constraints = []
    root_solution = compute_solution(agents, root_constraints, starts, goals)
    if root_solution is None:
        return None
    root_cost = compute_cost(root_solution)
    root = CBSNode(root_constraints, root_solution, root_cost)

    queue = []
    heapq.heappush(queue, root)

    while queue:
        node = heapq.heappop(queue)
        conflict = detect_conflict(node.solution)
        if not conflict:
            return node.solution

        for agent in [conflict['a1'], conflict['a2']]:
            new_constraints = list(node.constraints)

            if conflict['type'] == 'vertex':
                new_constraints.append({
                    'agent': agent,
                    'loc': conflict['loc'],
                    'time': conflict['time']
                })

            elif conflict['type'] == 'edge':
                if agent == conflict['a1']:
                    from_pos, to_pos = conflict['loc']
                else:
                    to_pos, from_pos = conflict['loc']
                new_constraints.append({
                    'agent': agent,
                    'loc': (from_pos, to_pos),
                    'time': conflict['time']
                })

            new_solution = compute_solution(agents, new_constraints, starts, goals)
            if new_solution:
                cost = compute_cost(new_solution)
                new_node = CBSNode(new_constraints, new_solution, cost)
                heapq.heappush(queue, new_node)
    return None

# =============================================================================
# ROS2 COLLISION AVOIDANCE NODE
# =============================================================================

class CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('collision_avoidance')
        
        # Publishers for robot control
        self.robot1_pub = self.create_publisher(Twist, '/robomaster_1/cmd_vel', 10)
        self.robot2_pub = self.create_publisher(Twist, '/robomaster_2/cmd_vel', 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_callback)  # 10 Hz
        
        # Robot state tracking
        self.robot_positions = {
            0: torch.tensor([starts[0]], dtype=torch.float32),
            1: torch.tensor([starts[1]], dtype=torch.float32)
        }
        self.robot_goals = {
            0: torch.tensor([goals[0]], dtype=torch.float32),
            1: torch.tensor([goals[1]], dtype=torch.float32)
        }
        
        # Path planning
        self.waypoint_plans = {}
        self.plan_paths()
        
        # Control parameters
        self.control_frequency = 10.0  # Hz
        self.max_linear_vel = 1  # m/s
        self.max_angular_vel = 1.0  # rad/s
        
        self.get_logger().info('Collision Avoidance Node started')
        self.get_logger().info(f'Robot 1: Start {starts[0]} -> Goal {goals[0]}')
        self.get_logger().info(f'Robot 2: Start {starts[1]} -> Goal {goals[1]}')
    
    def plan_paths(self):
        """Plan collision-free paths for both robots"""
        agents = [0, 1]
        starts_dict = {0: tuple(starts[0]), 1: tuple(starts[1])}
        goals_dict = {0: tuple(goals[0]), 1: tuple(goals[1])}
        
        self.get_logger().info('Planning collision-free paths...')
        plan = cbs(agents, starts_dict, goals_dict)
        
        if plan is None:
            self.get_logger().error('No valid path found!')
            return
        
        self.waypoint_plans = {k: list(v) for k, v in plan.items()}
        
        self.get_logger().info('Path planning completed!')
        for k, v in self.waypoint_plans.items():
            self.get_logger().info(f'Robot {k+1}: {len(v)} waypoints')
    
    def control_callback(self):
        """Main control loop - runs at 10 Hz"""
        if not self.waypoint_plans:
            return
        
        # Calculate forces for each robot
        for robot_id in [0, 1]:
            if robot_id not in self.waypoint_plans or len(self.waypoint_plans[robot_id]) == 0:
                continue
            
            # Get current position and next waypoint
            current_pos = self.robot_positions[robot_id][0]
            next_waypoint = self.waypoint_plans[robot_id][0]
            next_pos = torch.tensor([next_waypoint], dtype=torch.float32)
            
            # Calculate error vector
            error = next_pos - current_pos
            
            # Check if close enough to waypoint
            if torch.norm(error) < following_distance and len(self.waypoint_plans[robot_id]) > 1:
                self.waypoint_plans[robot_id].pop(0)
                self.get_logger().info(f'Robot {robot_id+1} reached waypoint: {next_waypoint}')
                continue
            
            # Calculate attractive force
            attractive_force = kp * error
            
            # Calculate repulsive force from other robot
            repulsive_force = torch.zeros_like(attractive_force)
            other_robot_id = 1 - robot_id  # Get the other robot's ID
            
            if other_robot_id in self.robot_positions:
                other_pos = self.robot_positions[other_robot_id][0]
                vec = current_pos - other_pos
                dist = torch.norm(vec)
                
                if dist < avoid_radius and dist > 1e-6:
                    repulsive_force += repulse_strength * vec / (dist**2)
            
            # Combine forces
            if len(self.waypoint_plans[robot_id]) == 1:
                force = attractive_force + repulsive_force
            else:
                if torch.norm(error) != 0:
                    force = error / torch.norm(error) * 1.0
                else:
                    force = torch.zeros_like(error)
            
            # Cap force
            force_norm = torch.norm(force)
            if force_norm > max_force:
                force = force / force_norm * max_force
            
            # Convert force to Twist message
            twist = Twist()
            
            # Convert force to linear velocity (scaled appropriately)
            force_x = force[0][0].item()
            force_y = force[0][1].item()
            
            # Scale force to velocity (you may need to adjust these scaling factors)
            linear_x = force_x * 2.0  # Scale factor for x-direction
            linear_y = force_y * 2.0  # Scale factor for y-direction
            
            # Limit velocities
            linear_x = 2 * max(-self.max_linear_vel, min(self.max_linear_vel, linear_x))
            linear_y = 2 * max(-self.max_linear_vel, min(self.max_linear_vel, linear_y))
            
            twist.linear.x = linear_x
            twist.linear.y = linear_y
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            # Publish to appropriate robot
            if robot_id == 0:
                self.robot1_pub.publish(twist)
            else:
                self.robot2_pub.publish(twist)
            
            # Log force vector
            self.get_logger().info(f'Robot {robot_id+1} force: ({force_x:.3f}, {force_y:.3f}) -> vel: ({linear_x:.3f}, {linear_y:.3f})')
            
            # Update robot position (simplified - in real implementation, you'd get this from odometry)
            # For simulation purposes, we'll update based on the velocity
            dt = 1.0 / self.control_frequency
            self.robot_positions[robot_id][0] += torch.tensor([linear_x * dt, linear_y * dt])
    
    def check_goals_reached(self):
        """Check if all robots have reached their goals"""
        for robot_id in [0, 1]:
            if robot_id in self.robot_positions and robot_id in self.robot_goals:
                current_pos = self.robot_positions[robot_id][0]
                goal_pos = self.robot_goals[robot_id][0]
                distance = torch.norm(current_pos - goal_pos)
                
                if distance < 0.1:  # Within 10cm of goal
                    self.get_logger().info(f'Robot {robot_id+1} reached goal!')
                    return True
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    
    node = CollisionAvoidanceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
