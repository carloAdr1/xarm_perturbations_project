import math
import numpy as np
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# ====== CONFIG ======
BAG_DIR = "rosbag2_2026_02_23-20_24_54"
BASE_FRAME = "link_base"
EEF_FRAME  = "link_eef"

TOPIC_DES = "/xarm/desired_point"
TOPIC_CMD = "/servo_server/delta_twist_cmds"
TOPIC_TF  = "/tf"

def read_topic(bag_dir, topic_name, msg_type_str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    msg_type = get_message(msg_type_str)
    out = []
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(raw, msg_type)
        t = t_ns * 1e-9
        out.append((t, msg))
    return out

def main():
    # --- Read messages ---
    des_msgs = read_topic(BAG_DIR, TOPIC_DES, "geometry_msgs/msg/PointStamped")
    cmd_msgs = read_topic(BAG_DIR, TOPIC_CMD, "geometry_msgs/msg/TwistStamped")
    tf_msgs  = read_topic(BAG_DIR, TOPIC_TF,  "tf2_msgs/msg/TFMessage")

    if not des_msgs:
        raise RuntimeError("No desired_point messages found in bag.")
    if not cmd_msgs:
        raise RuntimeError("No delta_twist_cmds messages found in bag.")
    if not tf_msgs:
        raise RuntimeError("No /tf messages found in bag.")

    # --- Build TF time series for BASE->EEF translation ---
    tf_times = []
    tf_xyz = []

    for t, tfm in tf_msgs:
        for tr in tfm.transforms:
            if tr.header.frame_id == BASE_FRAME and tr.child_frame_id == EEF_FRAME:
                tf_times.append(t)
                tf_xyz.append([tr.transform.translation.x,
                               tr.transform.translation.y,
                               tr.transform.translation.z])

    tf_times = np.array(tf_times, dtype=float)
    tf_xyz = np.array(tf_xyz, dtype=float)

    if tf_times.size == 0:
        raise RuntimeError(f"No TF found for {BASE_FRAME} -> {EEF_FRAME}. "
                           f"Check frame names or record /tf_static too if needed.")

    # --- Desired arrays (use desired timestamps as main timeline) ---
    t_des = np.array([t for t, _ in des_msgs], dtype=float)
    x_des = np.array([m.point.x for _, m in des_msgs], dtype=float)
    y_des = np.array([m.point.y for _, m in des_msgs], dtype=float)
    z_des = np.array([m.point.z for _, m in des_msgs], dtype=float)

    # Normalize time to start at zero (nice plots)
    t0 = t_des[0]
    t_des0 = t_des - t0
    tf_times0 = tf_times - t0

    # --- Interpolate actual pose from TF onto desired timeline ---
    # (TF is ~10 Hz, desired ~50 Hz -> interpolation is expected)
    # Note: requires tf_times sorted
    order = np.argsort(tf_times0)
    tf_times0 = tf_times0[order]
    tf_xyz = tf_xyz[order]

    x_act = np.interp(t_des0, tf_times0, tf_xyz[:,0])
    y_act = np.interp(t_des0, tf_times0, tf_xyz[:,1])
    z_act = np.interp(t_des0, tf_times0, tf_xyz[:,2])

    # --- Error over time ---
    ex = x_des - x_act
    ey = y_des - y_act
    ez = z_des - z_act
    e_norm = np.sqrt(ex**2 + ey**2 + ez**2)

    # --- Commanded velocity magnitude (separate timeline) ---
    t_cmd = np.array([t for t, _ in cmd_msgs], dtype=float) - t0
    vx = np.array([m.twist.linear.x for _, m in cmd_msgs], dtype=float)
    vy = np.array([m.twist.linear.y for _, m in cmd_msgs], dtype=float)
    vz = np.array([m.twist.linear.z for _, m in cmd_msgs], dtype=float)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # ====== PLOTS REQUIRED ======
    # 1) Desired vs actual position
    plt.figure()
    plt.plot(t_des0, x_des, label="x_des")
    plt.plot(t_des0, x_act, label="x_act")
    plt.plot(t_des0, y_des, label="y_des")
    plt.plot(t_des0, y_act, label="y_act")
    plt.plot(t_des0, z_des, label="z_des")
    plt.plot(t_des0, z_act, label="z_act")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Desired vs Actual Position (EEF)")
    plt.grid(True)
    plt.legend()

    # 2) Error over time (norm + components)
    plt.figure()
    plt.plot(t_des0, e_norm, label="||e||")
    plt.plot(t_des0, ex, label="e_x")
    plt.plot(t_des0, ey, label="e_y")
    plt.plot(t_des0, ez, label="e_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Tracking Error Over Time")
    plt.grid(True)
    plt.legend()

    # 3) Commanded velocity magnitude
    plt.figure()
    plt.plot(t_cmd, v_mag, label="|v_cmd|")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Commanded Velocity Magnitude")
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
