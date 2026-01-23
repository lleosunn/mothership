# Copyright 2011 Brown University Robotics.
# Copyright 2017 Open Source Robotics Foundation, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import threading

import geometry_msgs.msg
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


msg = """
This node takes keypresses from the keyboard and publishes them
as Twist/TwistStamped messages. It works best with a US keyboard layout.
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
    'i': (1, 0, 0, 0),
    'o': (1, 0, 0, -1),
    'j': (0, 0, 0, -1),
    'l': (0, 0, 0, 1),
    'u': (1, 0, 0, 1),
    ',': (-1, 0, 0, 0),
    '.': (-1, 0, 0, 1),
    'm': (-1, 0, 0, -1),
    'O': (1, -1, 0, 0),
    'I': (1, 0, 0, 0),
    'J': (0, -1, 0, 0),
    'L': (0, 1, 0, 0),
    'U': (1, 1, 0, 0),
    '<': (-1, 0, 0, 0),
    '>': (-1, -1, 0, 0),
    'M': (-1, 1, 0, 0),
    't': (0, 0, 1, 0),
    'b': (0, 0, -1, 0),
}

speedBindings = {
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}


def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(speed, turn):
    return 'currently:\tspeed %s\tturn %s ' % (speed, turn)


# add near the top
STOP_KEYS = {' ', 'k'}  # space or 'k' clears all active movement

def main():
    import time
    settings = saveTerminalSettings()
    rclpy.init()
    node = rclpy.create_node('teleop_twist_keyboard')

    # parameters
    stamped      = node.declare_parameter('stamped', False).value
    frame_id     = node.declare_parameter('frame_id', '').value
    repeat_rate  = float(node.declare_parameter('repeat_rate', 20.0).value)   # Hz
    topic1        = node.declare_parameter('topic1', '/robomaster_1/cmd_vel').value
    topic2        = node.declare_parameter('topic2', '/robomaster_2/cmd_vel').value
    TwistMsg = geometry_msgs.msg.TwistStamped if stamped else geometry_msgs.msg.Twist
    pub1 = node.create_publisher(TwistMsg, topic1, 10)
    pub2 = node.create_publisher(TwistMsg, topic2, 10)

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    # single mutable state (no nonlocal)
    state = {
        "speed": 0.5,
        "turn":  1.0,
        "active_keys": set(),        # << sticky set of movement keys
        "twist_msg": TwistMsg(),
    }
    lock = threading.Lock()

    def build_twist_from_active():
        # sum contributions from all latched movement keys
        x = y = z = th = 0.0
        for k in list(state["active_keys"]):
            if k in moveBindings:
                dx, dy, dz, dth = moveBindings[k]
                x += dx; y += dy; z += dz; th += dth
            else:
                state["active_keys"].discard(k)

        # clip to [-1, 1] so combos don't exceed 1
        def clip(v): return max(-1.0, min(1.0, v))
        x, y, z, th = map(clip, (x, y, z, th))

        tm = state["twist_msg"]
        if stamped:
            twist = tm.twist
            tm.header.stamp = node.get_clock().now().to_msg()
            if frame_id:
                tm.header.frame_id = frame_id
        else:
            twist = tm

        twist.linear.x  = x * state["speed"]
        twist.linear.y  = y * state["speed"]
        twist.linear.z  = z * state["speed"]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = th * state["turn"]

    # steady publisher
    period = 1.0 / max(1e-6, repeat_rate)
    def timer_cb():
        with lock:
            build_twist_from_active()
            pub1.publish(state["twist_msg"])
            pub2.publish(state["twist_msg"])
    node.create_timer(period, timer_cb)

    try:
        print(msg)
        print(vels(state["speed"], state["turn"]))
        while True:
            key = getKey(settings)
            with lock:
                if key in moveBindings:
                    # latch: treat as constantly “pressed”
                    state["active_keys"].add(key)
                elif key in STOP_KEYS:
                    # clear all motion
                    state["active_keys"].clear()
                elif key in speedBindings:
                    s_gain, t_gain = speedBindings[key]
                    state["speed"] *= s_gain
                    state["turn"]  *= t_gain
                    print(vels(state["speed"], state["turn"]))
                elif key == '\x03':  # CTRL-C
                    break
                # else: ignore other keys
    except Exception as e:
        print(e)
    finally:
        with lock:
            state["active_keys"].clear()
            build_twist_from_active()
            pub.publish(state["twist_msg"])
        rclpy.shutdown()
        spinner.join()
        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()
