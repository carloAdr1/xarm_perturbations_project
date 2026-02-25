#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PointStamped
from tf2_ros import Buffer, TransformListener


class PositionController(Node):
    def __init__(self):
        super().__init__("position_controller")

        self.output_topic = str(self.declare_parameter("output_topic", "/xarm/controller_cmd").value)

        kp = self.declare_parameter("kp", [2.5, 2.5, 2.5]).value
        kd = self.declare_parameter("kd", [0.6, 0.6, 0.6]).value
        ki = self.declare_parameter("ki", [0.0, 0.0, 0.0]).value

        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.ki = np.array(ki, dtype=float)

        self.max_speed = float(self.declare_parameter("max_speed", 0.10).value)
        db = self.declare_parameter("deadband", 0.002).value
        self.deadband = float(db)

        self.antiwindup = float(self.declare_parameter("antiwindup", 0.05).value)  # clamp integral term
        self.rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)

        self.pub = self.create_publisher(TwistStamped, self.output_topic, 10)
        self.sub = self.create_subscription(PointStamped, "/xarm/desired_point", self.on_desired, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.desired = None
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        self.prev_time = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)
        self.get_logger().info(f"position_controller OUT={self.output_topic} IN=/xarm/desired_point")

    def on_desired(self, msg: PointStamped):
        self.desired = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)

    def read_eef(self):
        try:
            tr = self.tf_buffer.lookup_transform("link_base", "link_eef", rclpy.time.Time())
            return np.array([tr.transform.translation.x,
                             tr.transform.translation.y,
                             tr.transform.translation.z], dtype=float)
        except Exception:
            return None

    def loop(self):
        if self.desired is None:
            return
        current = self.read_eef()
        if current is None:
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1e-6

        e = self.desired - current

        # deadband per-axis
        e_db = np.where(np.abs(e) < self.deadband, 0.0, e)

        de = (e_db - self.prev_error) / dt

        # integral (only if PID)
        if np.any(self.ki != 0.0):
            self.integral += e_db * dt
            # anti-windup clamp
            self.integral = np.clip(self.integral, -self.antiwindup, self.antiwindup)
        else:
            self.integral[:] = 0.0

        v = self.kp * e_db + self.kd * de + self.ki * self.integral

        # saturate by magnitude (not per-axis)
        speed = float(np.linalg.norm(v))
        if speed > self.max_speed and speed > 1e-9:
            v = v * (self.max_speed / speed)

        out = TwistStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "link_base"
        out.twist.linear.x = float(v[0])
        out.twist.linear.y = float(v[1])
        out.twist.linear.z = float(v[2])
        out.twist.angular.x = 0.0
        out.twist.angular.y = 0.0
        out.twist.angular.z = 0.0
        self.pub.publish(out)

        self.prev_error = e_db
        self.prev_time = now


def main():
    rclpy.init()
    n = PositionController()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
