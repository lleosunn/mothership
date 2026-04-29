import heapq
from itertools import combinations
import math
import time as _time

grid_scale_factor = 5  # world boundary in grid cells

# DJI RoboMaster EP Core: 320 × 240 mm
ROBOT_LENGTH_M = 0.320
ROBOT_WIDTH_M = 0.240
ROBOT_DIAGONAL_M = math.sqrt(ROBOT_LENGTH_M**2 + ROBOT_WIDTH_M**2)  # 0.40 m
SAFETY_FACTOR = 1.2
MIN_SAFE_DISTANCE_M = ROBOT_DIAGONAL_M * SAFETY_FACTOR  # 0.48 m


def min_separation_for_cell_size(cell_size_m: float) -> float:
    """Return the minimum separation in grid-cell units for a given cell size."""
    return MIN_SAFE_DISTANCE_M / cell_size_m


def _min_segment_dist(a0, a1, b0, b1) -> float:
    """Min Euclidean distance between two agents moving linearly in one timestep.

    Each agent travels a straight line from its start to its end position.
    Returns the closest the two line segments get at any point in [0, 1].
    """
    dx = a0[0] - b0[0]
    dy = a0[1] - b0[1]
    vx = (a1[0] - a0[0]) - (b1[0] - b0[0])
    vy = (a1[1] - a0[1]) - (b1[1] - b0[1])
    vv = vx * vx + vy * vy
    if vv < 1e-12:
        return math.sqrt(dx * dx + dy * dy)
    s = max(0.0, min(1.0, -(dx * vx + dy * vy) / vv))
    rx = dx + s * vx
    ry = dy + s * vy
    return math.sqrt(rx * rx + ry * ry)


class CBSNode:
    def __init__(self, constraints, solution, cost):
        self.constraints = constraints
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def astar(agent, start, goal, constraints, max_time=None):
    """A* search that respects vertex and edge constraints for the given agent."""
    if max_time is None:
        max_time = 4 * grid_scale_factor * 2 + 20

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
    nodes_expanded = 0

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        nodes_expanded += 1
        time = g

        if current == goal:
            print(f"  [A*] agent {agent}: {start}→{goal} solved in {nodes_expanded} nodes, "
                  f"path len={len(path)}, constraints={len(vertex_constraints)}v+{len(edge_constraints)}e")
            return path

        if time >= max_time:
            continue

        key = (current, time)
        if key in best_g and g >= best_g[key]:
            continue
        best_g[key] = g

        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            if not (-grid_scale_factor <= nx <= grid_scale_factor
                    and -grid_scale_factor <= ny <= grid_scale_factor):
                continue
            new_time = time + 1

            if (next_pos, new_time) in vertex_constraints:
                continue
            if (current, next_pos, new_time) in edge_constraints:
                continue

            new_g = g + 1
            new_f = new_g + h(next_pos)
            heapq.heappush(open_list, (new_f, new_g, next_pos, path + [next_pos]))

    print(f"  [A*] agent {agent}: {start}→{goal} FAILED after {nodes_expanded} nodes "
          f"(max_time={max_time}, constraints={len(vertex_constraints)}v+{len(edge_constraints)}e)")
    return None


def detect_conflict(paths, min_separation=0.0):
    """Detect the first conflict between agent paths.

    min_separation: minimum Euclidean distance (in grid cells) between any two
        agents at any instant.  0 = point-particle mode (original behaviour).
    """
    max_time = max(len(p) for p in paths.values())
    agents = list(paths.keys())

    for t in range(max_time):
        pos = {a: paths[a][min(t, len(paths[a]) - 1)] for a in agents}

        # --- vertex / proximity conflicts at time t ---
        for a1, a2 in combinations(agents, 2):
            p1, p2 = pos[a1], pos[a2]
            if min_separation > 0:
                d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if d < min_separation:
                    return {
                        'type': 'vertex', 'time': t,
                        'a1': a1, 'a2': a2,
                        'a1_loc': p1, 'a2_loc': p2,
                    }
            elif p1 == p2:
                return {
                    'type': 'vertex', 'time': t,
                    'a1': a1, 'a2': a2,
                    'a1_loc': p1, 'a2_loc': p2,
                }

        # --- edge / swept conflicts between t and t+1 ---
        if t + 1 < max_time:
            pos_next = {a: paths[a][min(t + 1, len(paths[a]) - 1)] for a in agents}
            for a1, a2 in combinations(agents, 2):
                p1t, p1n = pos[a1], pos_next[a1]
                p2t, p2n = pos[a2], pos_next[a2]

                if min_separation > 0:
                    d = _min_segment_dist(p1t, p1n, p2t, p2n)
                    if d < min_separation:
                        return {
                            'type': 'edge', 'time': t + 1,
                            'a1': a1, 'a2': a2,
                            'a1_loc': (p1t, p1n),
                            'a2_loc': (p2t, p2n),
                        }
                elif p1t == p2n and p2t == p1n:
                    return {
                        'type': 'edge', 'time': t + 1,
                        'a1': a1, 'a2': a2,
                        'a1_loc': (p1t, p1n),
                        'a2_loc': (p2t, p2n),
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


def cbs_fallback(agents, starts, goals):
    """CBS with point-particle mode (no swept-distance). Always fast to solve."""
    print("[CBS-fallback] Trying point-particle CBS (min_sep=0)...")
    root_solution = compute_solution(agents, [], starts, goals)
    if root_solution is None:
        return None
    root = CBSNode([], root_solution, compute_cost(root_solution))

    queue = [root]
    iterations = 0

    while queue and iterations < 2000:
        iterations += 1
        node = heapq.heappop(queue)
        conflict = detect_conflict(node.solution, 0.0)
        if not conflict:
            print(f"[CBS-fallback] SOLVED in {iterations} iterations, cost={node.cost}")
            return node.solution

        for agent in [conflict['a1'], conflict['a2']]:
            new_constraints = list(node.constraints)
            loc_key = 'a1_loc' if agent == conflict['a1'] else 'a2_loc'
            new_constraints.append({
                'agent': agent,
                'loc': conflict[loc_key],
                'time': conflict['time'],
            })
            new_solution = compute_solution(agents, new_constraints, starts, goals)
            if new_solution:
                heapq.heappush(queue, CBSNode(new_constraints, new_solution, compute_cost(new_solution)))

    print(f"[CBS-fallback] Failed after {iterations} iterations.")
    return None


def cbs(agents, starts, goals, min_separation=0.0, max_iterations=500, timeout_s=10.0):
    """Conflict-Based Search.

    min_separation: minimum distance in grid cells between any two agents
        at any instant (including mid-step).  0 = point particles.
    max_iterations: hard cap on CBS tree nodes explored.
    timeout_s: wall-clock timeout in seconds.
    """
    t0 = _time.monotonic()
    print(f"[CBS] Starting: {len(agents)} agents, min_sep={min_separation:.2f}, "
          f"max_iter={max_iterations}, timeout={timeout_s}s")
    print(f"[CBS]   starts={starts}")
    print(f"[CBS]   goals ={goals}")

    root_constraints = []
    root_solution = compute_solution(agents, root_constraints, starts, goals)
    if root_solution is None:
        print("[CBS] Root A* failed — no initial solution.")
        return None
    root_cost = compute_cost(root_solution)
    root = CBSNode(root_constraints, root_solution, root_cost)
    print(f"[CBS] Root solution cost={root_cost}")

    queue = []
    heapq.heappush(queue, root)
    iterations = 0
    best_node = root

    while queue:
        iterations += 1
        elapsed = _time.monotonic() - t0

        if iterations > max_iterations or elapsed > timeout_s:
            remaining_conflicts = detect_conflict(best_node.solution, min_separation)
            if remaining_conflicts:
                print(f"[CBS] TIMEOUT after {iterations} iterations, {elapsed:.2f}s, "
                      f"queue={len(queue)}. Best-effort still has conflicts — "
                      f"falling back to point-particle mode (no swept checks).")
                fallback = cbs_fallback(agents, starts, goals)
                if fallback:
                    return fallback
                print("[CBS] Fallback also failed. Returning None.")
                return None
            print(f"[CBS] TIMEOUT after {iterations} iterations, {elapsed:.2f}s. "
                  f"Best-effort is conflict-free (cost={best_node.cost}).")
            return best_node.solution

        node = heapq.heappop(queue)
        conflict = detect_conflict(node.solution, min_separation)

        if iterations % 50 == 0 or iterations <= 5:
            print(f"[CBS] iter={iterations}, queue={len(queue)}, "
                  f"cost={node.cost}, constraints={len(node.constraints)}, "
                  f"elapsed={elapsed:.2f}s")

        if not conflict:
            elapsed = _time.monotonic() - t0
            print(f"[CBS] SOLVED in {iterations} iterations, {elapsed:.2f}s, cost={node.cost}")
            return node.solution

        if node.cost <= best_node.cost:
            conflict_count_new = len([
                c for c in [detect_conflict(node.solution, min_separation)] if c
            ])
            conflict_count_old = len([
                c for c in [detect_conflict(best_node.solution, min_separation)] if c
            ])
            if conflict_count_new <= conflict_count_old:
                best_node = node

        if iterations <= 10:
            print(f"[CBS] conflict: {conflict['type']} t={conflict['time']} "
                  f"agents {conflict['a1']}@{conflict['a1_loc']} vs "
                  f"{conflict['a2']}@{conflict['a2_loc']}")

        for agent in [conflict['a1'], conflict['a2']]:
            new_constraints = list(node.constraints)
            loc_key = 'a1_loc' if agent == conflict['a1'] else 'a2_loc'

            if conflict['type'] == 'vertex':
                new_constraints.append({
                    'agent': agent,
                    'loc': conflict[loc_key],
                    'time': conflict['time'],
                })
            elif conflict['type'] == 'edge':
                new_constraints.append({
                    'agent': agent,
                    'loc': conflict[loc_key],
                    'time': conflict['time'],
                })

            new_solution = compute_solution(agents, new_constraints, starts, goals)
            if new_solution:
                cost = compute_cost(new_solution)
                new_node = CBSNode(new_constraints, new_solution, cost)
                heapq.heappush(queue, new_node)

    elapsed = _time.monotonic() - t0
    print(f"[CBS] EXHAUSTED queue after {iterations} iterations, {elapsed:.2f}s — no solution.")
    return None


if __name__ == '__main__':
    agents = [1, 2]
    starts = {1: (-10, -6), 2: (-2, -2)}
    goals = {1: (-2, 8), 2: (6, -4)}

    solution = cbs(agents, starts, goals)
    if solution:
        for agent, path in solution.items():
            print(f"Agent {agent}: {path}")
    else:
        print("No solution found.")
