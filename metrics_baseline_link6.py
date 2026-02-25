import sys
import numpy as np
from collections import defaultdict, deque

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

BAG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/bot1/xarm_ws/bags/baseline"
BASE_FRAME="link_base"
EEF_FRAME ="link6"

TOPIC_DES="/xarm/desired_point"
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

    nodes=[]
    cur=eef
    while cur!=base:
        nodes.append(cur)
        cur=prev[cur]
    nodes.reverse()

    T=np.eye(4, dtype=float)
    for n in nodes:
        T = T @ prevT[n]
    return T

def main():
    des = read_topic(TOPIC_DES,"geometry_msgs/msg/PointStamped")
    tfm = read_topic(TOPIC_TF, "tf2_msgs/msg/TFMessage")
    if not des: raise RuntimeError("No /xarm/desired_point in bag")
    if not tfm: raise RuntimeError("No /tf in bag")

    t0 = des[0][0]
    t_des = np.array([t-t0 for t,_ in des], dtype=float)
    x_des = np.array([m.point.x for _,m in des], dtype=float)
    y_des = np.array([m.point.y for _,m in des], dtype=float)
    z_des = np.array([m.point.z for _,m in des], dtype=float)

    tfm.sort(key=lambda x:x[0])
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

    mask = np.isfinite(x_act) & np.isfinite(y_act) & np.isfinite(z_act)
    if mask.sum() < 10:
        raise RuntimeError(f"Too few valid actual samples: {mask.sum()}/{len(mask)}")

    ex = x_des[mask] - x_act[mask]
    ey = y_des[mask] - y_act[mask]
    ez = z_des[mask] - z_act[mask]

    # RMSE per axis
    rmse_x = np.sqrt(np.mean(ex**2))
    rmse_y = np.sqrt(np.mean(ey**2))
    rmse_z = np.sqrt(np.mean(ez**2))

    # Total RMSE (vector)
    rmse_total = np.sqrt(np.mean(ex**2 + ey**2 + ez**2))

    # Max absolute position error (norm)
    e_norm = np.sqrt(ex**2 + ey**2 + ez**2)
    max_abs_pos_err = np.max(e_norm)

    # (extra) max abs per axis (por si lo piden)
    max_abs_x = np.max(np.abs(ex))
    max_abs_y = np.max(np.abs(ey))
    max_abs_z = np.max(np.abs(ez))

    print("=== Baseline metrics (using link_base -> link6) ===")
    print(f"Bag: {BAG_DIR}")
    print(f"Valid samples: {mask.sum()}")
    print("")
    print(f"RMSE_x (m):     {rmse_x:.6f}")
    print(f"RMSE_y (m):     {rmse_y:.6f}")
    print(f"RMSE_z (m):     {rmse_z:.6f}")
    print(f"Total RMSE (m): {rmse_total:.6f}")
    print("")
    print(f"Max |pos error| (norm) (m): {max_abs_pos_err:.6f}")
    print(f"Max |e_x| (m):             {max_abs_x:.6f}")
    print(f"Max |e_y| (m):             {max_abs_y:.6f}")
    print(f"Max |e_z| (m):             {max_abs_z:.6f}")

if __name__ == "__main__":
    main()
