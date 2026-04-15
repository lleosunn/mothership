#!/usr/bin/env python3
"""
Swap controller: CBS swap plan + cardinal-only motion in bursts with 2 s dwells.

- Zeroth phase: each robot aligns to the center of its CBS start cell using only
  east/west/north/south steps (Manhattan in world, one axis per primitive).
- Then: for each synchronized CBS index, both robots execute one cardinal step
  toward their next cell center together.
- Publishes /robomaster_{id}/cmd_vel (same body convention as move_to_goal).
- Logs every primordial move to the terminal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist

from .cbs import cbs
from . import cbs as cbs_module

Dir = Literal['east', 'west', 'north', 'south']


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class SinglePrim:
    robot: int
    direction: Dir
    target_wx: float
    target_wy: float


@dataclass
class PairPrim:
    step_index: int
    r1_move: bool
    r1_dir: Dir
    r1_tx: float
    r1_ty: float
    r2_move: bool
    r2_dir: Dir
    r2_tx: float
    r2_ty: float


def _grid_delta_to_dir(di: int, dj: int) -> Dir:
    if di == 1 and dj == 0:
        return 'east'
    if di == -1 and dj == 0:
        return 'west'
    if di == 0 and dj == 1:
        return 'north'
    if di == 0 and dj == -1:
        return 'south'
    raise ValueError(f'non-cardinal grid delta {(di, dj)}')


def _manhattan_primitives(
    cx: float,
    cy: float,
    gi: int,
    gj: int,
    ox: float,
    oy: float,
    cell: float,
) -> list[tuple[Dir, float, float]]:
    """Steps from (cx,cy) to cell center of (gi,gj), horizontal first then vertical."""
    Tx = ox + float(gi) * cell
    Ty = oy + float(gj) * cell
    out: list[tuple[Dir, float, float]] = []
    eps = 1e-4

    sign_x = 1.0 if Tx >= cx else -1.0
    while abs(Tx - cx) > eps:
        step = min(cell, abs(Tx - cx))
        nx = cx + sign_x * step
        d: Dir = 'east' if sign_x > 0 else 'west'
        out.append((d, nx, cy))
        cx = nx

    sign_y = 1.0 if Ty >= cy else -1.0
    while abs(Ty - cy) > eps:
        step = min(cell, abs(Ty - cy))
        ny = cy + sign_y * step
        d = 'north' if sign_y > 0 else 'south'
        out.append((d, cx, ny))
        cy = ny

    return out


class SwapMotion(Node):
    def __init__(self):
        super().__init__('swap_motion')

        self.declare_parameter('cell_size_m', 0.25)
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('dwell_between_primitives_s', 2.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('kp_linear', 1.8)
        self.declare_parameter('max_linear', 0.22)
        self.declare_parameter('position_close_m', 0.06)
        self.declare_parameter('primitive_timeout_s', 18.0)
        self.declare_parameter('plan_poll_s', 0.5)

        self._cell = float(self.get_parameter('cell_size_m').value)
        self._ox = float(self.get_parameter('origin_x').value)
        self._oy = float(self.get_parameter('origin_y').value)
        self._dwell = float(self.get_parameter('dwell_between_primitives_s').value)
        self._kp = float(self.get_parameter('kp_linear').value)
        self._vmax = float(self.get_parameter('max_linear').value)
        self._close = float(self.get_parameter('position_close_m').value)
        self._pto = float(self.get_parameter('primitive_timeout_s').value)

        self._fused: dict[int, PoseStamped | None] = {1: None, 2: None}
        self._cmd_pubs = {
            1: self.create_publisher(Twist, '/robomaster_1/cmd_vel', 10),
            2: self.create_publisher(Twist, '/robomaster_2/cmd_vel', 10),
        }

        self.create_subscription(PoseStamped, '/fused_pose_1', self._cb1, 10)
        self.create_subscription(PoseStamped, '/fused_pose_2', self._cb2, 10)

        self._queue: list[SinglePrim | PairPrim] = []
        self._qi = 0
        self._phase = 'idle'  # idle until plan; then dwell | move | done
        self._dwell_rem = 0.0
        self._active: SinglePrim | PairPrim | None = None
        self._prim_start = None
        self._planned = False

        period = 1.0 / float(self.get_parameter('control_rate').value)
        self.create_timer(period, self._control_tick)

        poll = float(self.get_parameter('plan_poll_s').value)
        self.create_timer(poll, self._try_plan)

        self.get_logger().info(
            f'swap_motion: cardinal swap + {self._dwell}s dwells; '
            f'cell={self._cell} m, CBS ±{cbs_module.grid_scale_factor}'
        )

    def _cb1(self, msg: PoseStamped):
        self._fused[1] = msg

    def _cb2(self, msg: PoseStamped):
        self._fused[2] = msg

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(round((x - self._ox) / self._cell))
        gy = int(round((y - self._oy) / self._cell))
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return self._ox + float(gx) * self._cell, self._oy + float(gy) * self._cell

    def _in_bounds(self, gx: int, gy: int) -> bool:
        lim = cbs_module.grid_scale_factor
        return -lim <= gx <= lim and -lim <= gy <= lim

    def _try_plan(self):
        if self._planned:
            return
        if self._fused[1] is None or self._fused[2] is None:
            return

        p1 = self._fused[1].pose
        p2 = self._fused[2].pose
        x1, y1 = p1.position.x, p1.position.y
        x2, y2 = p2.position.x, p2.position.y
        s1 = self._world_to_grid(x1, y1)
        s2 = self._world_to_grid(x2, y2)

        if not self._in_bounds(*s1) or not self._in_bounds(*s2):
            self.get_logger().warn(
                f'starts {s1}, {s2} outside CBS ±{cbs_module.grid_scale_factor}; adjust origin/cell.'
            )
            return
        if s1 == s2:
            self.get_logger().warn('robots in same grid cell; separate them.')
            return

        sol = cbs([1, 2], {1: s1, 2: s2}, {1: s2, 2: s1})
        if not sol:
            self.get_logger().error('CBS: no swap solution.')
            return

        max_len = max(len(sol[a]) for a in (1, 2))
        pad: dict[int, list[tuple[int, int]]] = {}
        for a in (1, 2):
            path = list(sol[a])
            last = path[-1]
            if len(path) < max_len:
                path = path + [last] * (max_len - len(path))
            pad[a] = path

        g1i, g1j = pad[1][0]
        g2i, g2j = pad[2][0]

        q: list[SinglePrim | PairPrim] = []

        self.get_logger().info('--- swap_motion: building primitive queue ---')
        self.get_logger().info(f'CBS padded paths len={max_len} agent1={pad[1]} agent2={pad[2]}')

        for pr in _manhattan_primitives(x1, y1, g1i, g1j, self._ox, self._oy, self._cell):
            d, tx, ty = pr
            q.append(SinglePrim(1, d, tx, ty))
        for pr in _manhattan_primitives(x2, y2, g2i, g2j, self._ox, self._oy, self._cell):
            d, tx, ty = pr
            q.append(SinglePrim(2, d, tx, ty))

        self.get_logger().info(
            f'Zeroth: align robot1 to cell center ({g1i},{g1j}), then robot2 to ({g2i},{g2j}).'
        )

        for k in range(max_len - 1):
            a1, b1 = pad[1][k], pad[1][k + 1]
            a2, b2 = pad[2][k], pad[2][k + 1]
            di1, dj1 = b1[0] - a1[0], b1[1] - a1[1]
            di2, dj2 = b2[0] - a2[0], b2[1] - a2[1]
            idle1 = di1 == 0 and dj1 == 0
            idle2 = di2 == 0 and dj2 == 0
            if idle1 and idle2:
                continue
            if idle1:
                t1x, t1y = self._grid_to_world(*a1)
                d1: Dir = 'east'
            else:
                t1x, t1y = self._grid_to_world(*b1)
                d1 = _grid_delta_to_dir(di1, dj1)
            if idle2:
                t2x, t2y = self._grid_to_world(*a2)
                d2: Dir = 'east'
            else:
                t2x, t2y = self._grid_to_world(*b2)
                d2 = _grid_delta_to_dir(di2, dj2)
            q.append(
                PairPrim(
                    k,
                    not idle1,
                    d1,
                    t1x,
                    t1y,
                    not idle2,
                    d2,
                    t2x,
                    t2y,
                )
            )

        self._queue = q
        self._qi = 0
        self._dwell_rem = 0.0
        self._active = None
        self._prim_start = None
        self._planned = True
        self._phase = 'dwell'
        self.get_logger().info(
            f'Planned {len(q)} primitives (zeroth align + {max_len - 1} synchronized CBS steps).'
        )

    def _pose_xy_yaw(self, rid: int) -> tuple[float, float, float] | None:
        m = self._fused[rid]
        if m is None:
            return None
        p = m.pose
        return p.position.x, p.position.y, yaw_from_quat(p.orientation)

    def _world_to_body_vw(self, vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        vx_b = c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w
        return vx_b, vy_b

    def _clamp_v(self, vx_b: float, vy_b: float) -> tuple[float, float]:
        lin_scale = 1.0
        if abs(vx_b) > 1e-9:
            lin_scale = min(lin_scale, self._vmax / abs(vx_b))
        if abs(vy_b) > 1e-9:
            lin_scale = min(lin_scale, self._vmax / abs(vy_b))
        return vx_b * lin_scale, vy_b * lin_scale

    def _twist_for_world_v(self, vx_w: float, vy_w: float, yaw: float) -> Twist:
        vx_b, vy_b = self._world_to_body_vw(vx_w, vy_w, yaw)
        vx_b, vy_b = self._clamp_v(vx_b, vy_b)
        t = Twist()
        t.linear.x = float(vx_b)
        t.linear.y = float(-vy_b)
        t.angular.z = 0.0
        return t

    def _dir_to_world_velocity(
        self, direction: Dir, cx: float, cy: float, tx: float, ty: float
    ) -> tuple[float, float]:
        ex, ey = tx - cx, ty - cy
        if direction in ('east', 'west'):
            return self._kp * ex, 0.0
        return 0.0, self._kp * ey

    def _at_target(self, cx: float, cy: float, tx: float, ty: float) -> bool:
        return math.hypot(tx - cx, ty - cy) < self._close

    def publish_stop_all(self):
        z = Twist()
        self._cmd_pubs[1].publish(z)
        self._cmd_pubs[2].publish(z)

    def _log_prim(self, msg: str):
        self.get_logger().info(msg)
        print(msg, flush=True)

    def _start_next(self):
        if self._qi >= len(self._queue):
            self._phase = 'done'
            self.publish_stop_all()
            self._log_prim('swap_motion: queue finished; cmd_vel zeroed for both robots.')
            return
        self._active = self._queue[self._qi]
        self._prim_start = self.get_clock().now()
        self._phase = 'move'
        a = self._active
        if isinstance(a, SinglePrim):
            self._log_prim(
                f'[swap_motion] PRIMITIVE start single robot={a.robot} dir={a.direction} '
                f'target=({a.target_wx:.4f},{a.target_wy:.4f}) queue_index={self._qi}'
            )
        else:
            self._log_prim(
                f'[swap_motion] PRIMITIVE start SYNC step={a.step_index} '
                f'r1 move={a.r1_move} dir={a.r1_dir} target=({a.r1_tx:.4f},{a.r1_ty:.4f}) '
                f'r2 move={a.r2_move} dir={a.r2_dir} target=({a.r2_tx:.4f},{a.r2_ty:.4f}) '
                f'queue_index={self._qi}'
            )

    def _control_tick(self):
        if not self._planned:
            return
        if self._phase == 'done':
            return

        if self._phase == 'dwell':
            self.publish_stop_all()
            dt = 1.0 / float(self.get_parameter('control_rate').value)
            self._dwell_rem -= dt
            if self._dwell_rem <= 0.0:
                self._start_next()
            return

        # move
        assert self._active is not None
        now = self.get_clock().now()
        if self._prim_start is not None:
            elapsed = (now - self._prim_start).nanoseconds / 1e9
            if elapsed > self._pto:
                self._log_prim(
                    f'[swap_motion] PRIMITIVE timeout after {elapsed:.1f}s; advancing. '
                    f'queue_index={self._qi}'
                )
                self.publish_stop_all()
                self._qi += 1
                self._phase = 'dwell'
                self._dwell_rem = self._dwell
                self._active = None
                return

        if isinstance(self._active, SinglePrim):
            rid = self._active.robot
            oth = 2 if rid == 1 else 1
            st = self._pose_xy_yaw(rid)
            if st is None:
                self.publish_stop_all()
                return
            cx, cy, yaw = st
            d = self._active.direction
            vx_w, vy_w = self._dir_to_world_velocity(d, cx, cy, self._active.target_wx, self._active.target_wy)
            tw = self._twist_for_world_v(vx_w, vy_w, yaw)
            self._cmd_pubs[rid].publish(tw)
            self._cmd_pubs[oth].publish(Twist())

            if self._at_target(cx, cy, self._active.target_wx, self._active.target_wy):
                self._log_prim(
                    f'[swap_motion] PRIMITIVE done single robot={rid} dir={d} '
                    f'at=({cx:.4f},{cy:.4f}) queue_index={self._qi}'
                )
                self.publish_stop_all()
                self._qi += 1
                self._phase = 'dwell'
                self._dwell_rem = self._dwell
                self._active = None
            return

        # PairPrim
        a = self._active
        st1 = self._pose_xy_yaw(1)
        st2 = self._pose_xy_yaw(2)
        if st1 is None or st2 is None:
            self.publish_stop_all()
            return
        c1x, c1y, y1 = st1
        c2x, c2y, y2 = st2
        if a.r1_move:
            v1w = self._dir_to_world_velocity(a.r1_dir, c1x, c1y, a.r1_tx, a.r1_ty)
            self._cmd_pubs[1].publish(self._twist_for_world_v(v1w[0], v1w[1], y1))
        else:
            self._cmd_pubs[1].publish(Twist())
        if a.r2_move:
            v2w = self._dir_to_world_velocity(a.r2_dir, c2x, c2y, a.r2_tx, a.r2_ty)
            self._cmd_pubs[2].publish(self._twist_for_world_v(v2w[0], v2w[1], y2))
        else:
            self._cmd_pubs[2].publish(Twist())

        ok1 = (not a.r1_move) or self._at_target(c1x, c1y, a.r1_tx, a.r1_ty)
        ok2 = (not a.r2_move) or self._at_target(c2x, c2y, a.r2_tx, a.r2_ty)
        if ok1 and ok2:
            self._log_prim(
                f'[swap_motion] PRIMITIVE done SYNC step={a.step_index} '
                f'r1=({c1x:.4f},{c1y:.4f}) r2=({c2x:.4f},{c2y:.4f}) queue_index={self._qi}'
            )
            self.publish_stop_all()
            self._qi += 1
            self._phase = 'dwell'
            self._dwell_rem = self._dwell
            self._active = None


def main(args=None):
    rclpy.init(args=args)
    node = SwapMotion()
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
