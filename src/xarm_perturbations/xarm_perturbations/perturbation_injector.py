#!/usr/bin/env python3
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class PerturbationInjector(Node):
    def __init__(self):
        super().__init__("perturbation_injector")

        self.input_topic = str(self.declare_parameter("input_topic", "/xarm/controller_cmd").value)
        self.output_topic = str(self.declare_parameter("output_topic", "/servo_server/delta_twist_cmds").value)

        self.enabled = bool(self.declare_parameter("enabled", True).value)
        self.mode = str(self.declare_parameter("mode", "off").value).lower().strip()  # off|sine|gaussian

        self.max_lin = float(self.declare_parameter("max_linear_speed", 0.20).value)

        # Sine
        self.sine_freq_hz = float(self.declare_parameter("sine_freq_hz", 1.0).value)
        self.sine_amp_linear = float(self.declare_parameter("sine_amp_linear", 0.01).value)
        self.sine_axis = str(self.declare_parameter("sine_axis", "x").value).lower().strip()

        # Gaussian
        self.gauss_std_linear = float(self.declare_parameter("gauss_std_linear", 0.01).value)
        self.gauss_axis = str(self.declare_parameter("gauss_axis", "x").value).lower().strip()

        self.debug = bool(self.declare_parameter("debug", True).value)
        self.debug_period_s = float(self.declare_parameter("debug_period_s", 1.0).value)
        self._last_dbg_wall = time.time()

        qos_name = str(self.declare_parameter("pub_reliability", "reliable").value).lower().strip()
        reliability = ReliabilityPolicy.RELIABLE if qos_name in ("reliable", "r") else ReliabilityPolicy.BEST_EFFORT

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(TwistStamped, self.output_topic, self.qos)
        self.sub = self.create_subscription(TwistStamped, self.input_topic, self.on_cmd, self.qos)

        self.t0 = time.time()
        self.rng = np.random.default_rng(7)

        self.get_logger().info(
            "✅ perturbation_injector\n"
            f"   IN : {self.input_topic}\n"
            f"   OUT: {self.output_topic}\n"
            f"   mode={self.mode} enabled={self.enabled}"
        )

    def _perturb(self):
        if (not self.enabled) or self.mode == "off":
            return np.zeros(3, dtype=float)

        axis_map = {"x": 0, "y": 1, "z": 2}
        ax = axis_map.get(self.sine_axis, 0)
        gx = axis_map.get(self.gauss_axis, 0)

        if self.mode == "sine":
            s = math.sin(2.0 * math.pi * self.sine_freq_hz * (time.time() - self.t0))
            dp = np.zeros(3, dtype=float)
            dp[ax] = self.sine_amp_linear * s
            return dp

        if self.mode == "gaussian":
            dp = np.zeros(3, dtype=float)
            dp[gx] = float(self.rng.normal(0.0, self.gauss_std_linear))
            return dp

        return np.zeros(3, dtype=float)

    def on_cmd(self, msg: TwistStamped):
        u = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=float)
        dp = self._perturb()
        v = np.clip(u + dp, -self.max_lin, self.max_lin)

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id if msg.header.frame_id else "link_base"
        out.twist.linear.x = float(v[0])
        out.twist.linear.y = float(v[1])
        out.twist.linear.z = float(v[2])
        out.twist.angular.x = 0.0
        out.twist.angular.y = 0.0
        out.twist.angular.z = 0.0

        self.pub.publish(out)

        if self.debug:
            now = time.time()
            if now - self._last_dbg_wall >= self.debug_period_s:
                self.get_logger().info(f"[dbg] u={np.round(u,3)} dp={np.round(dp,3)} out={np.round(v,3)} mode={self.mode}")
                self._last_dbg_wall = now


def main():
    rclpy.init()
    n = PerturbationInjector()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
