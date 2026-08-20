from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="sonar_mapping",
            executable="sonar_mapping_node",
            name="sonar_mapping_node",
            output="screen",
            parameters=[{
                "sonar_topic": "/oculus/sonar_image",
                "odom_topic": "/odometry",
                "crop_row": 200,
                "sonar_pitch_deg": 20.0,
                "akaze_threshold": 0.001,
                "map_publish_every": 20,
                "save_path": "~/sonar_mapping_map.ply",
            }],
        ),
    ])
