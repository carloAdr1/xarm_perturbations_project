import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

BAG_DIR="rosbag2_2026_02_23-20_24_54"
BASE_FRAME="link_base"
EEF_FRAME ="link6"   # <- clave: en tu bag solo llega hasta link6

TOPIC_DES="/xarm/desired_point"
TOPIC_CMD="/servo_server/delta_twist_cmds"
TOPIC_TF="/tf"

def quat_to_rot(x,y,z,w):
    xx,yy,zz=x*x,y*y,z*z
    xy,xz,yz=x*y,x*z,y*z
    wx,wy,wz=w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

def tf_to_T(tr):
    p = tr.transform.translation
    q = tr.transform.rotation
    T = np.eye(4, dtype=float)
    T[:3,:3] = quat_to_rot(q.x,q.y,q.z,q.w)
    T[:3,3]  = [p.x,p.y,p.z]
    return T

def invT(T):
    R=T[:3,:3]; t=T[:3,3]
    Ti=np.eye(4, dtype=float)
    Ti[:3,:3]=R.T
    Ti[:3,3]=-(R.T@t)
    return Ti

def read_topic(topic, msg_type_str):
    reader=SequentialReader()
    reader.open(
        StorageOptions(uri=BAG_DIR, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    MsgT=get_message(msg_type_str)
    out=[]
    while reader.has_next():
        tp, raw, t_ns = reader.read_next()
        if tp!=topic: 
            continue
        out.append((t_ns*1e-9, deserialize_message(raw, MsgT)))
    out.sort(key=lambda x:x[0])
    return out

def compose(latest_edges, base, eef):
    graph=defaultdict(list)
    for (p,c),Tpc in latest_edges.items():
        graph[p].append((c,Tpc))
        graph[c].append((p,invT(Tpc)))

    q=deque([base])
    prev={base:None}
    prevT={}
    while q:
        u=q.popleft()
        if u==eef: break
        for v,Tuv in graph[u]:
            if v in prev: 
                continue
            prev[v]=u
            prevT[v]=Tuv
            q.append(v)

    if eef not in prev:
        return None

    # reconstruct
    nodes=[]
    cur=eef
    while cur!=base:
        nodes.append(cur)
        cur=prev[cur]
    nodes.reverse()

    T=np.eye(4, dtype=float)
    cur=base
    for n in nodes:
        T = T @ prevT[n]
        cur=n
    return T

def main():
    des = read_topic(TOPIC_DES,"geometry_msgs/msg/PointStamped")
    cmd = read_topic(TOPIC_CMD,"geometry_msgs/msg/TwistStamped")
    tfm = read_topic(TOPIC_TF, "tf2_msgs/msg/TFMessage")

    t0 = des[0][0]
    t_des = np.array([t-t0 for t,_ in des], dtype=float)
    x_des = np.array([m.point.x for _,m in des], dtype=float)
    y_des = np.array([m.point.y for _,m in des], dtype=float)
    z_des = np.array([m.point.z for _,m in des], dtype=float)

    t_cmd = np.array([t-t0 for t,_ in cmd], dtype=float)
    vx = np.array([m.twist.linear.x for _,m in cmd], dtype=float)
    vy = np.array([m.twist.linear.y for _,m in cmd], dtype=float)
    vz = np.array([m.twist.linear.z for _,m in cmd], dtype=float)
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)

    tf_times = np.array([t-t0 for t,_ in tfm], dtype=float)

    latest_edges={}
    tf_idx=0

    x_act=np.full_like(t_des, np.nan, dtype=float)
    y_act=np.full_like(t_des, np.nan, dtype=float)
    z_act=np.full_like(t_des, np.nan, dtype=float)

    for i,td in enumerate(t_des):
        while tf_idx < len(tfm) and tf_times[tf_idx] <= td:
            _, msg = tfm[tf_idx]
            for tr in msg.transforms:
                p = tr.header.frame_id.lstrip("/")
                c = tr.child_frame_id.lstrip("/")
                latest_edges[(p,c)] = tf_to_T(tr)
            tf_idx += 1

        T = compose(latest_edges, BASE_FRAME, EEF_FRAME)
        if T is not None:
            x_act[i], y_act[i], z_act[i] = T[0,3], T[1,3], T[2,3]

    valid = np.isfinite(x_act).sum()
    print(f"[OK] Using TF chain: {BASE_FRAME} -> {EEF_FRAME}")
    print(f"[OK] Actual samples: {valid}/{len(t_des)}")

    mask = np.isfinite(x_act) & np.isfinite(y_act) & np.isfinite(z_act)

    ex = x_des[mask] - x_act[mask]
    ey = y_des[mask] - y_act[mask]
    ez = z_des[mask] - z_act[mask]
    enorm = np.sqrt(ex*ex + ey*ey + ez*ez)

    # 1) Desired vs actual
    plt.figure()
    plt.plot(t_des, x_des, label="x_des")
    plt.plot(t_des[mask], x_act[mask], label="x_act(link6)")
    plt.plot(t_des, y_des, label="y_des")
    plt.plot(t_des[mask], y_act[mask], label="y_act(link6)")
    plt.plot(t_des, z_des, label="z_des")
    plt.plot(t_des[mask], z_act[mask], label="z_act(link6)")
    plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
    plt.title("Desired vs Actual Position (base->link6)")
    plt.grid(True); plt.legend()
    plt.savefig("/home/bot1/xarm_logs/plot_des_vs_act.png", dpi=200)

    # 2) Error over time
    plt.figure()
    plt.plot(t_des[mask], enorm, label="||e||")
    plt.plot(t_des[mask], ex, label="e_x")
    plt.plot(t_des[mask], ey, label="e_y")
    plt.plot(t_des[mask], ez, label="e_z")
    plt.xlabel("Time (s)"); plt.ylabel("Error (m)")
    plt.title("Tracking Error Over Time (using link6 actual)")
    plt.grid(True); plt.legend()
    plt.savefig("/home/bot1/xarm_logs/plot_error.png", dpi=200)

    # 3) Commanded velocity magnitude
    plt.figure()
    plt.plot(t_cmd, vmag, label="|v_cmd|")
    plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
    plt.title("Commanded Velocity Magnitude")
    plt.grid(True); plt.legend()
    plt.savefig("/home/bot1/xarm_logs/plot_vmag.png", dpi=200)

    plt.show()

if __name__=="__main__":
    main()
