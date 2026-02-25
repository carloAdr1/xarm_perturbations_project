import matplotlib.pyplot as plt
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

BAG_DIR = "rosbag2_2026_02_23-20_24_54"

def read_topic(topic_name, msg_type_str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=BAG_DIR, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    msg_type = get_message(msg_type_str)
    t_list = []
    data = []
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(raw, msg_type)
            t = t_ns * 1e-9
            t_list.append(t)
            data.append(msg)
    return t_list, data

# Read desired trajectory
t_des, des = read_topic("/xarm/desired_point", "geometry_msgs/msg/PointStamped")

# Read commanded velocity
t_cmd, cmd = read_topic("/servo_server/delta_twist_cmds", "geometry_msgs/msg/TwistStamped")

# Extract data
x_des = [m.point.x for m in des]
y_des = [m.point.y for m in des]
z_des = [m.point.z for m in des]

vx_cmd = [m.twist.linear.x for m in cmd]
vy_cmd = [m.twist.linear.y for m in cmd]
vz_cmd = [m.twist.linear.z for m in cmd]

# Normalize time to start at zero
if t_des:
    t0 = t_des[0]
    t_des = [t - t0 for t in t_des]
if t_cmd:
    t0 = t_cmd[0]
    t_cmd = [t - t0 for t in t_cmd]

# Plot desired position
plt.figure()
plt.plot(t_des, x_des, label="x_des")
plt.plot(t_des, y_des, label="y_des")
plt.plot(t_des, z_des, label="z_des")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Desired Position")
plt.legend()
plt.grid()

# Plot commanded velocities
plt.figure()
plt.plot(t_cmd, vx_cmd, label="vx_cmd")
plt.plot(t_cmd, vy_cmd, label="vy_cmd")
plt.plot(t_cmd, vz_cmd, label="vz_cmd")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Commanded Velocity")
plt.legend()
plt.grid()

plt.show()
