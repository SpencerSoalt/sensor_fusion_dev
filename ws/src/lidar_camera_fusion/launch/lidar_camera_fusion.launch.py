from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        DeclareLaunchArgument(
            'calibration_file',
            default_value='/ws/lidar_camera_extrinsics.yaml',
            description='Path to camera-LiDAR calibration YAML',
        ),
        DeclareLaunchArgument(
            'max_distance', default_value='40.0',
            description='Max LiDAR projection distance (metres)',
        ),
        DeclareLaunchArgument(
            'min_distance', default_value='0.5',
            description='Min LiDAR projection distance (metres)',
        ),
        DeclareLaunchArgument(
            'sync_slop', default_value='0.05',
            description='ApproximateTimeSynchronizer slop (seconds)',
        ),

        Node(
            package='lidar_camera_fusion',
            executable='lidar_camera_fusion',
            name='lidar_camera_fusion',
            output='screen',
            parameters=[{
                'use_sim_time':     True,
                'calibration_file': LaunchConfiguration('calibration_file'),
                'max_distance':     LaunchConfiguration('max_distance'),
                'min_distance':     LaunchConfiguration('min_distance'),
                'sync_slop':        LaunchConfiguration('sync_slop'),
            }],
            remappings=[
                ('/pointcloud',    '/velodyne_points'),
                ('/detections_2d', '/yolo/detections'),
            ],
        ),

    ])
