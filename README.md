# Mothership

A ROS 2 workspace for a multi-agent robotics testbed. Two DJI RoboMaster robots are localized via overhead cameras using ArUco markers, fused through an Extended Kalman Filter, and driven by various control strategies including proportional control, repulsive avoidance, and Conflict-Based Search (CBS) multi-agent path planning.

## Repository Structure

```
src/
├── localization/          # Camera-based ArUco localization + EKF fusion
├── control/               # Motion control and path planning
└── aruco_marker_stuff/    # Standalone calibration & marker utilities
```

## Packages

### `localization`

Handles the full perception pipeline — from raw camera frames to fused robot poses.

**Nodes:**


| Node                    | Description                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera0_publisher`     | Captures frames from webcam 0 and publishes to `/camera0/image_raw`                                                                                                                                     |
| `camera1_publisher`     | Captures frames from webcam 1 and publishes to `/camera1/image_raw`                                                                                                                                     |
| `dual_camera_detection` | Detects ArUco markers in both camera feeds, establishes a world frame from marker 0, computes world-frame poses for robot markers (1–N), and fuses detections from both cameras. Publishes to `/pose_N` |
| `velocity_fusion`       | Subscribes to robot linear velocity and IMU attitude, computes angular velocity, and publishes fused body-frame twist to `/twist_N`                                                                     |
| `ekf`                   | Extended Kalman Filter fusing global pose observations (`/pose_N`) with body-frame velocities (`/twist_N`). Publishes smoothed pose to `/fused_pose_N`                                                  |
| `pose_visualizer`       | Real-time matplotlib 2D visualization of raw and fused poses for all agents                                                                                                                             |


**Launch file — `src/localization/launch/launch.xml`:**
Launches all 6 localization nodes (both cameras, detection, velocity fusion, EKF, and visualizer).

### `control`

Motion control nodes that subscribe to localized poses and drive the robots.

**Nodes:**


| Node                    | Description                                                                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `move_to_goal`          | **Main control node.** Subscribes to `/goal_pose_N` and `/fused_pose_N`, uses P-control in the robot body frame to drive each agent to its goal. Configurable gains, velocity limits, and tolerances. Stops if poses go stale (>1 s) |
| `simple_rotation`       | Test node that publishes a looping sequence of goal poses to `/goal_pose_1` and `/goal_pose_2`                                                                                                                                       |
| `cbs_scenario`          | Runs the CBS multi-agent path planner and publishes computed waypoints to `/goal_pose_N`                                                                                                                                             |
| `stay_in_place`         | P-controller that drives each agent to a fixed target position                                                                                                                                                                       |
| `repulsive_avoidance`   | Goal-seeking with inter-agent repulsive force avoidance. Press Enter to swap targets                                                                                                                                                 |
| `teleop_twist_keyboard` | Keyboard teleoperation for both robots simultaneously                                                                                                                                                                                |


**Support module — `cbs.py`:**
Implements Conflict-Based Search (CBS) for multi-agent pathfinding with A search, vertex/edge constraints, and conflict detection.

**Launch files:**


| File                               | Launches                           |
| ---------------------------------- | ---------------------------------- |
| `src/control/launch/launch.xml`    | `move_to_goal` + `simple_rotation` |
| `src/control/launch/cbslaunch.xml` | `move_to_goal` + `cbs_scenario`    |


### `aruco_marker_stuff`

Standalone OpenCV scripts (not a ROS 2 package) for ArUco marker generation and camera calibration.


| Script                 | Description                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `generate_marker.py`   | Generates ArUco markers 0–9 (DICT_6X6_250) as PNG images                                                             |
| `generate_charuco.py`  | Generates a ChArUco calibration board image                                                                          |
| `capture_charuco.py`   | Interactive webcam capture tool for collecting calibration images (press `s` to save, `q` to quit)                   |
| `calibrate_charuco.py` | Runs camera calibration from captured ChArUco images, saves results to `calibration.json`                            |
| `detection.py`         | Real-time ArUco detection and pose estimation — marker 0 defines the world frame, marker 1 is tracked relative to it |


Also contains pre-generated marker images (`marker_0.png`–`marker_9.png`), a ChArUco board image, captured calibration images in `charuco_images/`, and saved calibration data.

## Field Setup

1. **Turn on all robots** and ensure they are on the same network as the laptop. Verify by checking that their publishable topics are visible (e.g. via `ros2 topic list`).
2. **Plug cameras into the laptop** and calibrate them if not already calibrated. Use the scripts in `aruco_marker_stuff/`:
  - Print a ChArUco board with `generate_charuco.py`
  - Capture calibration images with `capture_charuco.py`
  - Run calibration with `calibrate_charuco.py`
3. **Place the calibration marker (marker 0) on the ground.** Both cameras must be able to see marker 0, as well as the entire area the robots will be operating in. Marker 0 defines the world frame origin.
4. **Launch the localization pipeline:**
  ```bash
   ros2 launch localization launch.xml
  ```
5. **Run the main control node** (`move_to_goal` in the `control` package) to drive robots to their goals:
  ```bash
   ros2 run control move_to_goal
  ```
   Or launch it alongside a goal source:

## Topic Flow

```
Physical Cameras
    ├─ camera0_publisher ──→ /camera0/image_raw ──┐
    └─ camera1_publisher ──→ /camera1/image_raw ──┤
                                                   ↓
                                    dual_camera_detection
                                           │
                                           └──→ /pose_N
                                                   │
    RoboMaster Robots                              │
    ├─ /robomaster_N/vel ──────┐                   │
    └─ /robomaster_N/attitude ─┤                   │
                               ↓                   │
                        velocity_fusion            │
                               │                   │
                               └──→ /twist_N       │
                                       │           │
                                       ↓           ↓
                                         ekf
                                          │
                                          └──→ /fused_pose_N
                                                    │
                             ┌──────────────────────┤
                             ↓                      ↓
                      pose_visualizer       move_to_goal
                                                    │
                                                    └──→ /robomaster_N/cmd_vel
                                                              │
                                                              ↓
                                                        Physical Robots

    Goal sources → /goal_pose_N → move_to_goal:
    ├─ simple_rotation   (cyclic test pattern)
    └─ cbs_scenario      (CBS-planned paths)
```

