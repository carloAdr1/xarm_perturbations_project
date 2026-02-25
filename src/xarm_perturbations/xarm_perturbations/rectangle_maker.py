#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener


def _vec_from_plane(plane: str):
    """
    Returns two unit axis vectors (u_axis, v_axis) in XYZ for a named plane.
    plane: 'xy', 'xz', 'yz'
    """
    plane = plane.lower()
    if plane == "xy":
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    if plane == "xz":
        return (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    if plane == "yz":
        return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    # default
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


class RectangleMaker(Node):
    """
    Publishes a rectangular Cartesian trajectory as PointStamped to /xarm/desired_point.

    Parameters:
      - radius (float): half-size scale. Rectangle will be width=2*radius, height=radius (can adjust below).
      - frequency (float): loops per second (Hz). Period = 1/frequency.
      - plane (str): 'xy', 'xz', 'yz'
      - base_frame (str): TF base frame (default: link_base)
      - eef_frame (str): TF end-effector frame (default: link_eef)
      - topic (str): output topic (default: /xarm/desired_point)
      - width_scale (float): width = 2*radius*width_scale (default: 1.0)
      - height_scale (float): height = radius*height_scale (default: 1.0)
      - rate_hz (float): publish rate (default: 50.0)
    """

    def __init__(self):
        super().__init__("rectangle_maker")

        self.declare_parameter("radius", 0.02)
        self.declare_parameter("frequency", 0.05)
        self.declare_parameter("plane", "xy")
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("topic", "/xarm/desired_point")
        self.declare_parameter("width_scale", 1.0)
        self.declare_parameter("height_scale", 1.0)
        self.declare_parameter("rate_hz", 50.0)

        self.radius = float(self.get_parameter("radius").value)
        self.freq = float(self.get_parameter("frequency").value)
        self.plane = str(self.get_parameter("plane").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.topic = str(self.get_parameter("topic").value)
        self.width_scale = float(self.get_parameter("width_scale").value)
        self.height_scale = float(self.get_parameter("height_scale").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        # Rectangle sizes
        self.width = 2.0 * self.radius * self.width_scale
        self.height = 1.0 * self.radius * self.height_scale

        self.pub = self.create_publisher(PointStamped, self.topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.center = None  # (cx, cy, cz)
        self.t0 = self.get_clock().now()

        dt = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(f"rectangle_maker ready -> {self.topic}")
        self.get_logger().info(
            f"params: radius={self.radius} freq={self.freq} plane={self.plane} "
            f"width={self.width:.3f} height={self.height:.3f}"
        )

    def _try_set_center_from_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.eef_frame, rclpy.time.Time()
            )
            self.center = (
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
                float(tf.transform.translation.z),
            )
            self.get_logger().info(f"Center set to [{self.center[0]:.3f} {self.center[1]:.3f} {self.center[2]:.3f}]")
            return True
        except Exception:
            return False

    @staticmethod
    def _lerp(a, b, s):
        return a + (b - a) * s

    def _rect_point(self, phase):
        """
        phase in [0,1) for one loop. We traverse 4 edges with constant speed.
        Rectangle corners (u,v): (+w/2,+h/2) -> (-w/2,+h/2) -> (-w/2,-h/2) -> (+w/2,-h/2) -> back
        """
        w2 = self.width * 0.5
        h2 = self.height * 0.5

        # Segment index 0..3, local s in [0,1)
        seg = int(phase * 4.0)
        s = (phase * 4.0) - seg

        if seg == 0:
            # top edge: (+w2,+h2) -> (-w2,+h2)
            u = self._lerp(+w2, -w2, s); v = +h2
        elif seg == 1:
            # left edge: (-w2,+h2) -> (-w2,-h2)
            u = -w2; v = self._lerp(+h2, -h2, s)
        elif seg == 2:
            # bottom edge: (-w2,-h2) -> (+w2,-h2)
            u = self._lerp(-w2, +w2, s); v = -h2
        else:
            # right edge: (+w2,-h2) -> (+w2,+h2)
            u = +w2; v = self._lerp(-h2, +h2, s)

        return u, v

    def _tick(self):
        if self.center is None:
            if not self._try_set_center_from_tf():
                return  # wait for TF

        # time -> phase
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        if self.freq <= 0.0:
            phase = 0.0
        else:
            period = 1.0 / self.freq
            phase = (t % period) / period

        u, v = self._rect_point(phase)
        u_axis, v_axis = _vec_from_plane(self.plane)

        cx, cy, cz = self.center
        x = cx + u * u_axis[0] + v * v_axis[0]
        y = cy + u * u_axis[1] + v * v_axis[1]
        z = cz + u * u_axis[2] + v * v_axis[2]

        msg = PointStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.base_frame
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = RectangleMaker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
