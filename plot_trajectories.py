#!/usr/bin/env python3
"""Animate 2D agent trajectories from a pose_log CSV file in real time.

Usage:
    python3 plot_trajectories.py <csv_file> [--speed 2.0] [--save]

Options:
    --speed N   Playback speed multiplier (default: 1.0 = real time)
    --save      Save animation as .mp4 instead of displaying live
"""
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path


AGENT_COLORS = {
    1: '#e6194b',
    2: '#3cb44b',
    3: '#4363d8',
    4: '#f58231',
    7: '#911eb4',
}

AGENT_LABELS = {
    1: 'Agent 1',
    2: 'Agent 2',
    3: 'Agent 3 (rm5)',
    4: 'Agent 4',
    7: 'Agent 7',
}

ROBOT_SIZE_M = 0.25


def load_csv(csv_path):
    """Load CSV into a dict of {agent_id: {t, x, y}} with numpy arrays."""
    raw = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = int(row['agent_id'])
            if aid not in raw:
                raw[aid] = {'t': [], 'x': [], 'y': []}
            raw[aid]['t'].append(float(row['timestamp']))
            raw[aid]['x'].append(float(row['x']))
            raw[aid]['y'].append(float(row['y']))

    for aid in raw:
        for k in ('t', 'x', 'y'):
            raw[aid][k] = np.array(raw[aid][k])
    return raw


def interp_at(data, t):
    """Interpolate agent position at global time t."""
    ts = data['t']
    if t <= ts[0]:
        return data['x'][0], data['y'][0]
    if t >= ts[-1]:
        return data['x'][-1], data['y'][-1]
    return float(np.interp(t, ts, data['x'])), float(np.interp(t, ts, data['y']))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--speed', type=float, default=3.0,
                        help='Playback speed multiplier (default: 3.0)')
    parser.add_argument('--save', action='store_true',
                        help='Save as .mp4 instead of live display')
    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f'File not found: {args.csv_file}')
        sys.exit(1)

    trajectories = load_csv(args.csv_file)
    if not trajectories:
        print('No data found in CSV.')
        sys.exit(1)

    agents = sorted(trajectories.keys())
    t_min = min(d['t'][0] for d in trajectories.values())
    t_max = max(d['t'][-1] for d in trajectories.values())
    duration = t_max - t_min

    all_x = np.concatenate([d['x'] for d in trajectories.values()])
    all_y = np.concatenate([d['y'] for d in trajectories.values()])
    pad = 0.3
    x_lo, x_hi = all_x.min() - pad, all_x.max() + pad
    y_lo, y_hi = all_y.min() - pad, all_y.max() + pad

    fps = 30
    n_frames = max(int(duration / args.speed * fps), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    trail_lines = {}
    robot_dots = {}
    robot_labels = {}
    trail_x = {a: [] for a in agents}
    trail_y = {a: [] for a in agents}

    for aid in agents:
        color = AGENT_COLORS.get(aid, '#999999')
        trail_lines[aid], = ax.plot([], [], color=color, linewidth=2, alpha=0.4)
        robot_dots[aid] = plt.Circle((0, 0), ROBOT_SIZE_M / 2, color=color,
                                     ec='black', linewidth=1.5, zorder=10)
        ax.add_patch(robot_dots[aid])
        robot_labels[aid] = ax.text(0, 0, str(aid), ha='center', va='center',
                                    fontsize=9, fontweight='bold', color='white',
                                    zorder=11)

    for aid in agents:
        x0, y0 = interp_at(trajectories[aid], t_min)
        ax.plot(x0, y0, 'o', color=AGENT_COLORS.get(aid, '#999'), markersize=8,
                markeredgecolor='black', markeredgewidth=1, alpha=0.4, zorder=4)

    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=12,
                        fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.8, edgecolor='#ccc'))

    legend_handles = []
    for aid in agents:
        color = AGENT_COLORS.get(aid, '#999999')
        label = AGENT_LABELS.get(aid, f'Agent {aid}')
        legend_handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.9)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=13)
    ax.set_ylabel('Y (m)', fontsize=13)
    ax.set_title(f'Trajectory Replay — {args.csv_file.name}', fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')

    def update(frame):
        frac = frame / max(n_frames - 1, 1)
        t_now = t_min + frac * duration

        for aid in agents:
            x, y = interp_at(trajectories[aid], t_now)
            trail_x[aid].append(x)
            trail_y[aid].append(y)
            trail_lines[aid].set_data(trail_x[aid], trail_y[aid])
            robot_dots[aid].center = (x, y)
            robot_labels[aid].set_position((x, y))

        elapsed = t_now - t_min
        time_text.set_text(f't = {elapsed:.1f}s / {duration:.1f}s  '
                           f'({args.speed:.1f}x)')
        return list(trail_lines.values()) + list(robot_dots.values()) + \
               list(robot_labels.values()) + [time_text]

    interval_ms = 1000 / fps
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms,
                         blit=False, repeat=False)

    if args.save:
        out_path = args.csv_file.with_suffix('.mp4')
        print(f'Saving animation to {out_path} ({n_frames} frames)...')
        anim.save(str(out_path), writer='ffmpeg', fps=fps, dpi=120)
        print(f'Done: {out_path}')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
