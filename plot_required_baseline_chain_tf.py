import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# ===== CONFIG =====
BAG_DIR = "rosbag2_2026_02_23-20_24_54"

# Ajusta si tus frames cambian (en tu launch se ve link_base y link_eef)
BASE_FRAME = "link_base"
EEF_FRAME  = "link_eef"

TOPIC_DES = "/xarm/desired_point"
TOPIC_CMD = "/servo_server/delta_twist_cmds"
TOPIC_TF  = "/tf"

def quat_to_rot(qx, qy, qz, qw):
    # Rotation matrix from quaternion (x,y,z,w)
    # Standard formula
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=float)
    return R

def tf_to_T(tr):
    # Build 4x4 homogeneous transform from geometry_msgs/TransformStamped
    tx = tr.transform.translation.x
    ty = tr.transform.translation.y
    tz = tr.transform.translation.z
    qx = tr.transform.rotation.x
    qy = tr.transform.rotation.y
    qz = tr.transform.rotation.z
    qw = tr.transform.rotation.w
    R = quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def invert_T(T):
    R = T[:3,:3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti

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

def compose_chain(latest_edges, base, eef):
    """
    latest_edges: dict[(parent,child)] = 4x4 T_parent_child
    We build a graph with both directions:
      parent -> child uses T_parent_child
      child  -> parent uses inverse(T_parent_child)
    Then find path base -> eef and multiply transforms along the path.
    """
    graph = defaultdict(list)
    for (p, c), Tpc in latest_edges.items():
        graph[p].append((c, Tpc))
        graph[c].append((p, invert_T(Tpc)))

    # BFS to find a path
    q = deque([base])
    prev = {base: None}
    prev_T = {}  # prev_T[node] = T_prev_to_node
    while q:
        u = q.popleft()
        if u == eef:
            break
        for v, Tuv in graph[u]:
            if v in prev:
                continue
            prev[v] = u
            prev_T[v] = Tuv
            q.append(v)

    if eef not in prev:
        return None  # no path

    # Reconstruct transform base->eef by walking back
    nodes = []
    cur = eef
    while cur != base:
        nodes.append(cur)
        cur = prev[cur]
    nodes.reverse()

    T = np.eye(4, dtype=float)
    cur = base
    for n in nodes:
        T = T @ prev_T[n]
        cur = n
    return T

def main():
    des_msgs = read_topic(BAG_DIR, TOPIC_DES, "geometry_msgs/msg/PointStamped")
    cmd_msgs = read_topic(BAG_DIR, TOPIC_CMD, "geometry_msgs/msg/TwistStamped")
    tf_msgs  = read_topic(BAG_DIR, TOPIC_TF,  "tf2_msgs/msg/TFMessage")

    if not des_msgs or not cmd_msgs or not tf_msgs:
        raise RuntimeError("Missing one or more topics in the bag.")

    # Sort by time (just in case)
    des_msgs.sort(key=lambda x: x[0])
    cmd_msgs.sort(key=lambda x: x[0])
    tf_msgs.sort(key=lambda x: x[0])

    t0 = des_msgs[0][0]

    # Desired timeline
    t_des = np.array([t - t0 for t, _ in des_msgs], dtype=float)
    x_des = np.array([m.point.x for _, m in des_msgs], dtype=float)
    y_des = np.array([m.point.y for _, m in des_msgs], dtype=float)
    z_des = np.array([m.point.z for _, m in des_msgs], dtype=float)

    # Commanded velocity magnitude
    t_cmd = np.array([t - t0 for t, _ in cmd_msgs], dtype=float)
    vx = np.array([m.twist.linear.x for _, m in cmd_msgs], dtype=float)
    vy = np.array([m.twist.linear.y for _, m in cmd_msgs], dtype=float)
    vz = np.array([m.twist.linear.z for _, m in cmd_msgs], dtype=float)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Build "latest TF buffer" while sweeping desired timestamps
    latest_edges = {}  # (parent,child)->T_parent_child
    tf_idx = 0

    x_act = np.full_like(t_des, np.nan, dtype=float)
    y_act = np.full_like(t_des, np.nan, dtype=float)
    z_act = np.full_like(t_des, np.nan, dtype=float)

    # Convert tf_msgs times to relative
    tf_times = [t - t0 for t, _ in tf_msgs]

    for i, td in enumerate(t_des):
        # Update TF buffer with all TF messages up to this desired time
        while tf_idx < len(tf_msgs) and tf_times[tf_idx] <= td:
            _, tfm = tf_msgs[tf_idx]
            for tr in tfm.transforms:
                parent = tr.header.frame_id
                child  = tr.child_frame_id
                latest_edges[(parent, child)] = tf_to_T(tr)
            tf_idx += 1

        Tbe = compose_chain(latest_edges, BASE_FRAME, EEF_FRAME)
        if Tbe is not None:
            x_act[i], y_act[i], z_act[i] = Tbe[0,3], Tbe[1,3], Tbe[2,3]

    # Error
    ex = x_des - x_act
    ey = y_des - y_act
    ez = z_des - z_act
    e_norm = np.sqrt(ex**2 + ey**2 + ez**2)

    # ===== PLOTS REQUIRED =====
    # 1) Desired vs actual position
    plt.figure()
    plt.plot(t_des, x_des, label="x_des")
    plt.plot(t_des, x_act, label="x_act")
    plt.plot(t_des, y_des, label="y_des")
    plt.plot(t_des, y_act, label="y_act")
    plt.plot(t_des, z_des, label="z_des")
    plt.plot(t_des, z_act, label="z_act")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Desired vs Actual Position ({BASE_FRAME} -> {EEF_FRAME})")
    plt.grid(True)
    plt.legend()

    # 2) Error over time
    plt.figure()
    plt.plot(t_des, e_norm, label="||e||")
    plt.plot(t_des, ex, label="e_x")
    plt.plot(t_des, ey, label="e_y")
    plt.plot(t_des, ez, label="e_z")
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

    # Quick sanity print
    valid = np.isfinite(x_act).sum()
    print(f"[OK] Actual pose samples computed: {valid}/{len(t_des)}")
    if valid == 0:
        print("WARNING: No actual samples computed. Frame names may not match BAG TF frames.")

    plt.show()

if __name__ == "__main__":
    main()
