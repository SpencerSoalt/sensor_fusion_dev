"""
Launch file for the LiDAR + 2D detection → 3D box fusion node.

Usage:
  ros2 launch lidar_bbox_fusion fusion.launch.py

Override any parameter:
  ros2 launch lidar_bbox_fusion fusion.launch.py cloud_topic:=/ouster/points max_range_m:=50.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Declare all launch arguments with defaults ----

    args = [
        # Topics
        DeclareLaunchArgument('cloud_topic',        default_value='/velodyne_points'),
        DeclareLaunchArgument('cam_info_topic',     default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('detections_topic',   default_value='/camera/detections'),

        # Frames
        DeclareLaunchArgument('lidar_frame',            default_value='velodyne'),
        DeclareLaunchArgument('camera_optical_frame',   default_value='camera_color_optical_frame'),
        DeclareLaunchArgument('target_frame',           default_value='base_link'),

        # Point cloud processing
        DeclareLaunchArgument('min_points_in_box',  default_value='10'),
        DeclareLaunchArgument('max_points_used',    default_value='60000'),
        DeclareLaunchArgument('stride',             default_value='2'),
        DeclareLaunchArgument('max_range_m',        default_value='30.0'),

        # Depth estimation
        DeclareLaunchArgument('publish_when_no_depth',  default_value='false'),
        DeclareLaunchArgument('fallback_depth_m',       default_value='15.0'),
        DeclareLaunchArgument('bbox_crop_ratio',        default_value='0.5'),

        # 3D box estimation
        DeclareLaunchArgument('use_lshape_fitting', default_value='true'),
        DeclareLaunchArgument('lshape_min_points',  default_value='25'),

        # Tracker
        DeclareLaunchArgument('enable_tracking',    default_value='false'),
        DeclareLaunchArgument('tracker_max_misses', default_value='5'),
        DeclareLaunchArgument('tracker_gate_m',     default_value='3.0'),
        DeclareLaunchArgument('min_hits_to_publish', default_value='2'),

        # Time sync
        DeclareLaunchArgument('sync_slop_s',        default_value='0.1'),
        DeclareLaunchArgument('sync_queue_size',    default_value='15'),

        # Visualization
        DeclareLaunchArgument('marker_alpha',       default_value='0.6'),
        DeclareLaunchArgument('marker_lifetime_s',  default_value='0.3'),
    ]

    # ---- Node ----

    detection_node = Node(
        package='detection_2_5d',
        executable='detection_2_5d',
        name='detection_2_5d',
        output='screen',
        parameters=[{
            # Topics
            'cloud_topic':              LaunchConfiguration('cloud_topic'),
            'cam_info_topic':           LaunchConfiguration('cam_info_topic'),
            'detections_topic':         LaunchConfiguration('detections_topic'),
            # Frames
            'lidar_frame':              LaunchConfiguration('lidar_frame'),
            'camera_optical_frame':     LaunchConfiguration('camera_optical_frame'),
            'target_frame':             LaunchConfiguration('target_frame'),
            # Point cloud
            'min_points_in_box':        LaunchConfiguration('min_points_in_box'),
            'max_points_used':          LaunchConfiguration('max_points_used'),
            'stride':                   LaunchConfiguration('stride'),
            'max_range_m':              LaunchConfiguration('max_range_m'),
            # Depth
            'publish_when_no_depth':    LaunchConfiguration('publish_when_no_depth'),
            'fallback_depth_m':         LaunchConfiguration('fallback_depth_m'),
            'bbox_crop_ratio':          LaunchConfiguration('bbox_crop_ratio'),
            # 3D box
            'use_lshape_fitting':       LaunchConfiguration('use_lshape_fitting'),
            'lshape_min_points':        LaunchConfiguration('lshape_min_points'),
            # Tracker
            'enable_tracking':          LaunchConfiguration('enable_tracking'),
            'tracker_max_misses':       LaunchConfiguration('tracker_max_misses'),
            'tracker_gate_m':           LaunchConfiguration('tracker_gate_m'),
            'min_hits_to_publish':      LaunchConfiguration('min_hits_to_publish'),
            # Sync
            'sync_slop_s':              LaunchConfiguration('sync_slop_s'),
            'sync_queue_size':          LaunchConfiguration('sync_queue_size'),
            # Viz
            'marker_alpha':             LaunchConfiguration('marker_alpha'),
            'marker_lifetime_s':        LaunchConfiguration('marker_lifetime_s'),
        }],
    )

    return LaunchDescription(args + [detection_node])
