from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_DIR="rosbag2_2026_02_23-20_24_54"

reader = SequentialReader()
reader.open(
    StorageOptions(uri=BAG_DIR, storage_id="sqlite3"),
    ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
)

TFMsg = get_message("tf2_msgs/msg/TFMessage")
edges=set()

while reader.has_next():
    topic, raw, t_ns = reader.read_next()
    if topic != "/tf":
        continue
    msg = deserialize_message(raw, TFMsg)
    for tr in msg.transforms:
        p = tr.header.frame_id.lstrip("/")
        c = tr.child_frame_id.lstrip("/")
        edges.add((p,c))

print(f"Unique TF edges: {len(edges)}")
for p,c in sorted(edges):
    print(f"{p} -> {c}")
