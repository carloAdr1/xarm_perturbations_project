#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener

class CircleServoXArmLite6(Node):
    def __init__(self):
        super().__init__("circle_servo_xarm_lite6")
        self.servo_pub = self.create_publisher(
            TwistStamped, "/servo_server/delta_twist_cmds", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.radius = 0.04
        self.frequency = 0.05
        self.plane = "xy"
        self.kp = np.array([2.5, 2.5, 2.5])
        self.kd = np.array([0.6, 0.6, 0.6])
        self.epsilon = np.array([0.002, 0.002, 0.002])
        self.max_speed = 0.10
        self.center = None
        self.start_time = self.get_clock().now()
        self.prev_error = np.zeros(3)
        self.prev_time = self.get_clock().now()
        self.last_info_time = self.get_clock().now()
        self.timer = self.create_timer(0.02, self._loop)
        self.get_logger().info("CircleServo iniciado. Esperando TF...")

    def _read_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "link_base", "link_eef", rclpy.time.Time())
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z], dtype=float)
        except Exception:
            return None

    def _circle_target(self, t):
        cx, cy, cz = self.center
        w = 2.0 * math.pi * self.frequency
        ramp = min(t / 2.0, 1.0)
        a = ramp * self.radius * math.cos(w * t)
        b = ramp * self.radius * math.sin(w * t)
        if self.plane == "xy":
            return np.array([cx + a, cy + b, cz])
        elif self.plane == "xz":
            return np.array([cx + a, cy, cz + b])
        else:
            return np.array([cx, cy + a, cz + b])

    def _loop(self):
        if self.center is None:
            p = self._read_pose()
            if p is None:
                return
            self.center = p.copy()
            self.start_time = self.get_clock().now()
            self.prev_time = self.get_clock().now()
            self.get_logger().info(f"Centro: {self.center.round(3)}")
            return

        current = self._read_pose()
        if current is None:
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1e-6

        t = (now - self.start_time).nanoseconds / 1e9
        target = self._circle_target(t)
        error = target - current
        d_error = (error - self.prev_error) / dt
        active = np.abs(error) > self.epsilon
        v = np.where(active, self.kp * error + self.kd * d_error, 0.0)
        v = np.clip(v, -self.max_speed, self.max_speed)

        msg = TwistStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "link_base"
        msg.twist.linear.x = float(v[0])
        msg.twist.linear.y = float(v[1])
        msg.twist.linear.z = float(v[2])
        self.servo_pub.publish(msg)

        self.prev_error = error
        self.prev_time = now

        if (now - self.last_info_time).nanoseconds > 1e9:
            self.get_logger().info(
                f"pos={current.round(3)} target={target.round(3)} v={v.round(3)}")
            self.last_info_time = now

def main(args=None):
    rclpy.init(args=args)
    node = CircleServoXArmLite6()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
