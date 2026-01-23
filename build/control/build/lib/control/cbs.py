import heapq
from collections import deque
import math

grid_scale_factor = 5 # not actually a scale factor, just a world boundary


class CBSNode:
    def __init__(self, constraints, solution, cost):
        self.constraints = constraints  # list: a constraint is a dict {'agent': agent, 'loc': (x, y), 'time': t}
        self.solution = solution        # dict: maps agents to paths
        self.cost = cost                # int: sum of path lengths for all agents combined

    def __lt__(self, other):
        """Compare nodes based on their cost for priority queue"""
        return self.cost < other.cost


def astar(agent, start, goal, constraints):
    """
    A* search that respects vertex and edge constraints for the given agent.

    Returns a list of positions [(x1, y1), ...] representing the path,
    or None if no path is found.
    """
    # --- Pre‑index constraints for O(1) checks ----------------------------
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

    # Manhattan heuristic
    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # (f, g, pos, path)
    open_list = [(h(start), 0, start, [start])]
    best_g = {}  # (pos, time) -> g

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        time = g  # one step = one time unit

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

            # Constraint checks
            if (next_pos, new_time) in vertex_constraints:
                continue
            if (current, next_pos, new_time) in edge_constraints:
                continue

            new_g = g + 1
            new_f = new_g + h(next_pos)
            heapq.heappush(open_list, (new_f, new_g, next_pos, path + [next_pos]))

    return None


def detect_conflict(paths):
    """
    paths: dict mapping agents to their paths, where each path is a list of positions

    returns a dict with conflict type ('vertex' or 'edge'), time step, involved agents, and location
    """
    max_time = max(len(p) for p in paths.values())
    for t in range(max_time):
        pos_at_t = {}
        for a, path in paths.items():
            pos = path[min(t, len(path) - 1)]
            if pos in pos_at_t:
                return {'type': 'vertex', 'time': t, 'a1': pos_at_t[pos], 'a2': a, 'loc': pos}
            pos_at_t[pos] = a

        # Check for edge conflict
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
    """
    agents: list of agent indices
    constraints: list of constraints for the CBS node
    starts: dict mapping agents to their starting positions
    goals: dict mapping agents to their goal positions

    returns a dict mapping agents to their paths or None if no solution found
    """
    solution = {}
    for agent in agents:
        path = astar(agent, starts[agent], goals[agent], constraints)
        if not path:
            return None
        solution[agent] = path
    return solution


def compute_cost(solution):
    """
    solution: dict mapping agents to their paths

    returns the total cost as the sum of path lengths for all agents
    """
    if solution is None:
        return float("inf")  # treat infeasible solutions as very costly
    # cost = number of moves (vertices – 1) so waiting at goal isn’t over‑penalised
    return sum(max(len(path) - 1, 0) for path in solution.values())


def cbs(agents, starts, goals):
    """
    agents: list of agent indices
    starts: dict mapping agents to their starting positions
    goals: dict mapping agents to their goal positions

    returns a dict mapping agents to their paths or None if no solution found
    """
    root_constraints = []
    root_solution = compute_solution(agents, root_constraints, starts, goals)
    if root_solution is None:
        # No feasible individual paths – exit early
        return None
    root_cost = compute_cost(root_solution)
    root = CBSNode(root_constraints, root_solution, root_cost)

    queue = []
    # orders the nodes by cost
    heapq.heappush(queue, root)

    while queue:
        # pop the node with the lowest cost
        node = heapq.heappop(queue)
        conflict = detect_conflict(node.solution)
        if not conflict:
            return node.solution

        # create branches for each agent involved in the conflict
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

# Example usage
if __name__ == '__main__':
    agents = [1, 2]
    starts = {
        1: (-10, -6),
        2: (-2, -2),
    }

    goals = {
        1: (-2, 8),
        2: (6, -4),
    }

    solution = cbs(agents, starts, goals)
    if solution:
        for agent, path in solution.items():
            print(f"Agent {agent}: {path}")
    else:
        print("No solution found.")
