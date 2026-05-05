import glob
import fcntl
import os
import struct

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# V4L2 ioctl to query device capabilities
_VIDIOC_QUERYCAP = 0x80685600
# Capability flag for single-planar video capture
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001


def _detect_capture_devices():
    """Return sorted list of /dev/videoN indices that support video capture."""
    indices = []
    for path in sorted(glob.glob('/dev/video*')):
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            try:
                buf = bytearray(104)
                fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
                # device_caps is a __u32 at offset 88 in struct v4l2_capability
                device_caps = struct.unpack_from('<I', buf, 88)[0]
                if device_caps & _V4L2_CAP_VIDEO_CAPTURE:
                    idx = int(path.replace('/dev/video', ''))
                    indices.append(idx)
            finally:
                os.close(fd)
        except OSError:
            continue
    return indices


def generate_launch_description():
    device_indices_arg = DeclareLaunchArgument(
        'device_indices',
        default_value='auto',
        description='Comma-separated OpenCV device indices (e.g. "0,1,4"), '
                    'or "auto" to detect all connected capture devices'
    )

    def _build_nodes(context):
        raw = LaunchConfiguration('device_indices').perform(context)

        if raw.strip() == 'auto':
            indices = _detect_capture_devices()
        else:
            indices = [int(x.strip()) for x in raw.split(',')]

        if not indices:
            return [LogInfo(msg='⚠️  No video capture devices found.')]

        num_cameras = len(indices)

        nodes = [
            LogInfo(msg=f'📷 Launching {num_cameras} camera(s) at device indices {indices}')
        ]
        for camera_id, device_index in enumerate(indices):
            nodes.append(Node(
                package='localization',
                executable='camera_publisher',
                name=f'camera{camera_id}_publisher',
                output='screen',
                parameters=[{
                    'camera_id': camera_id,
                    'device_index': device_index,
                }],
            ))

        nodes.append(Node(
            package='localization',
            executable='multi_camera_detection',
            name='multi_camera_detection',
            output='screen',
            parameters=[{
                'num_cameras': num_cameras,
            }],
        ))

        return nodes

    return LaunchDescription([
        device_indices_arg,
        OpaqueFunction(function=_build_nodes),
    ])
