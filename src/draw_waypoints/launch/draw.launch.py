from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('draw_waypoints')
    
    with open(os.path.join(pkg, 'config', 'robot.urdf'), 'r') as f:
        urdf = f.read()
    with open(os.path.join(pkg, 'config', 'robot.srdf'), 'r') as f:
        srdf = f.read()

    return LaunchDescription([
        Node(
            package='draw_waypoints',
            executable='draw_rectangle',
            parameters=[{
                'robot_description': urdf,
                'robot_description_semantic': srdf,
            }],
            output='screen',
            emulate_tty=True,
        )
    ])
