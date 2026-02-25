#!/usr/bin/env python3
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, TwistStamped
from tf2_msgs.msg import TFMessage


@dataclass
class Vec3:
    x: float
    y: float
    z: float


def now_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class DataLogger(Node):
    """
    Logs:
      - desired point: /xarm/desired_point (PointStamped)
      - actual ee pose from /tf (TFMessage): link_base -> link_eef (translation only)
      - commanded twist: /servo_server/delta_twist_cmds (TwistStamped)

    Output: CSV with synchronized-by-timestamp later in analysis.
    """

    def __init__(self):
        super().__init__("data_logger")

        self.declare_parameter("desired_topic", "/xarm/desired_point")
        self.declare_parameter("cmd_topic", "/servo_server/delta_twist_cmds")
        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("out_dir", os.path.expanduser("~/xarm_logs"))
        self.declare_parameter("experiment", "baseline")  # baseline | sine | gaussian

        self.desired_topic = str(self.get_parameter("desired_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.tf_topic = str(self.get_parameter("tf_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.experiment = str(self.get_parameter("experiment").value)

        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, f"{self.experiment}.csv")

        # cache latest values
        self.last_des: Optional[Tuple[float, Vec3]] = None
        self.last_cmd: Optional[Tuple[float, Vec3]] = None

        # open csv
        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t_actual",
            "x_act", "y_act", "z_act",
            "t_des", "x_des", "y_des", "z_des",
            "t_cmd", "vx_cmd", "vy_cmd", "vz_cmd",
        ])
        self.f.flush()

        # subs
        self.create_subscription(PointStamped, self.desired_topic, self.on_desired, 50)
        self.create_subscription(TwistStamped, self.cmd_topic, self.on_cmd, 50)
        self.create_subscription(TFMessage, self.tf_topic, self.on_tf, 50)

        self.get_logger().info(f"Logging to: {self.csv_path}")
        self.get_logger().info(f"DES: {self.desired_topic}  CMD: {self.cmd_topic}  TF: {self.tf_topic}")
        self.get_logger().info(f"Frames: {self.base_frame} -> {self.eef_frame}")

    def on_desired(self, msg: PointStamped):
        t = now_sec(msg.header.stamp)
        self.last_des = (t, Vec3(msg.point.x, msg.point.y, msg.point.z))

    def on_cmd(self, msg: TwistStamped):
        t = now_sec(msg.header.stamp)
        self.last_cmd = (t, Vec3(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z))

    def _extract_tf(self, tfmsg: TFMessage) -> Optional[Tuple[float, Vec3]]:
        # Find transform base->eef (translation only). TFMessage contains list of transforms.
        for tr in tfmsg.transforms:
            if tr.header.frame_id == self.base_frame and tr.child_frame_id == self.eef_frame:
                t = now_sec(tr.header.stamp)
                v = Vec3(tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z)
                return (t, v)
        return None

    def on_tf(self, msg: TFMessage):
        actual = self._extract_tf(msg)
        if actual is None:
            return

        t_act, v_act = actual
        # write row using latest desired & cmd (may be slightly different timestamp; analysis will resample properly)
        if self.last_des is None or self.last_cmd is None:
            return

        t_des, v_des = self.last_des
        t_cmd, v_cmd = self.last_cmd

        self.w.writerow([
            f"{t_act:.9f}",
            v_act.x, v_act.y, v_act.z,
            f"{t_des:.9f}", v_des.x, v_des.y, v_des.z,
            f"{t_cmd:.9f}", v_cmd.x, v_cmd.y, v_cmd.z
        ])
        # flush occasionally
        if int(t_act * 10) % 10 == 0:
            self.f.flush()

    def destroy_node(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    main()
