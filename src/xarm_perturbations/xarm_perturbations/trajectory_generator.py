#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__("trajectory_generator")

        self.pub = self.create_publisher(PointStamped, "/xarm/desired_point", 10)

        # Params
        self.ax = float(self.declare_parameter("ax", 0.06).value)   # m
        self.ay = float(self.declare_parameter("ay", 0.04).value)   # m
        self.fx = float(self.declare_parameter("fx", 0.08).value)   # Hz
        self.fy = float(self.declare_parameter("fy", 0.12).value)   # Hz
        self.phase = float(self.declare_parameter("phase", math.pi/2).value)
        self.hold_z = bool(self.declare_parameter("hold_z", True).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        self.soft_start_s = float(self.declare_parameter("soft_start_s", 2.0).value)

        # TF for choosing a safe center once
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.center = None

        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)

        self.get_logger().info("trajectory_generator ready -> /xarm/desired_point")

    def read_eef(self):
        try:
            tr = self.tf_buffer.lookup_transform("link_base", "link_eef", rclpy.time.Time())
            return np.array([tr.transform.translation.x,
                             tr.transform.translation.y,
                             tr.transform.translation.z], dtype=float)
        except Exception:
            return None

    def loop(self):
        if self.center is None:
            p = self.read_eef()
            if p is None:
                return
            self.center = p.copy()
            self.t0 = self.get_clock().now()
            self.get_logger().info(f"Center set to {np.round(self.center,3)}")
            return

        t = (self.get_clock().now() - self.t0).nanoseconds / 1e9

        # soft-start ramp
        r = min(max(t / self.soft_start_s, 0.0), 1.0)

        x = self.center[0] + r * self.ax * math.sin(2*math.pi*self.fx*t)
        y = self.center[1] + r * self.ay * math.sin(2*math.pi*self.fy*t + self.phase)
        z = self.center[2] if self.hold_z else (self.center[2] + r * 0.02 * math.sin(2*math.pi*0.07*t))

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "link_base"
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TrajectoryGenerator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
