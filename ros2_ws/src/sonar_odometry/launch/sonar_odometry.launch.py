"""
Launch file for the Sonar Inertial Odometry node.

Usage:
    ros2 launch sonar_odometry sonar_odometry.launch.py
    ros2 launch sonar_odometry sonar_odometry.launch.py \
        config:=/path/to/my_params.yaml

All parameters are read from the config YAML — edit that file to tune behaviour.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("sonar_odometry")
    default_config = PathJoinSubstitution([pkg_share, "config", "sonar_odometry.yaml"])

    return LaunchDescription([
        # ── Config file ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Path to YAML parameter file",
        ),

        # ── Nodes ──────────────────────────────────────────────────────────
        # NOTE: static transforms (imu → base_link, sonar → base_link) are
        # published by static_tf_mari_rov.launch.py — run that separately.
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter_madgwick_node",
            output="screen",
            parameters=[LaunchConfiguration("config")],
            remappings=[
                ("imu/data_raw", "/ouster/imu"),
                ("imu/data",     "/imu/data"),
            ],
        ),

        Node(
            package="sonar_odometry",
            executable="sonar_odometry_node",
            name="sonar_odometry_node",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),

        Node(
            package="sonar_odometry",
            executable="gps_path_node",
            name="gps_path_node",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),

        Node(
            package="sonar_odometry",
            executable="lidar_path_logger_node",
            name="lidar_path_logger_node",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),

        #Node(
        #    package="sonar_odometry",
        #    executable="imu_dr_node",
        #    name="imu_dr_node",
        #    output="screen",
        #    parameters=[
        #        LaunchConfiguration("config"),
        #        {"initial_heading_deg": ParameterValue(
        #            LaunchConfiguration("initial_heading_deg"), value_type=float
        #        )},
        #    ],
        #),
    ])
