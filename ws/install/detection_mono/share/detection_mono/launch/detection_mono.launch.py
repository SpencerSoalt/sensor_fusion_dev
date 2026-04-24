from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    args = [
        DeclareLaunchArgument('cam_info_topic',       default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('detections_topic',     default_value='/camera/detections'),
        DeclareLaunchArgument('camera_optical_frame', default_value='camera_color_optical_frame'),
        DeclareLaunchArgument('target_frame',         default_value='base_link'),

        DeclareLaunchArgument('min_box_height_px',      default_value='10'),
        DeclareLaunchArgument('max_depth_m',            default_value='80.0'),
        DeclareLaunchArgument('fallback_depth_m',       default_value='15.0'),
        DeclareLaunchArgument('use_aspect_ratio_yaw',   default_value='true'),

        DeclareLaunchArgument('enable_tracking',      default_value='true'),
        DeclareLaunchArgument('tracker_max_misses',   default_value='5'),
        DeclareLaunchArgument('tracker_gate_m',       default_value='5.0'),
        DeclareLaunchArgument('min_hits_to_publish',  default_value='2'),

        DeclareLaunchArgument('marker_alpha',         default_value='0.6'),
        DeclareLaunchArgument('marker_lifetime_s',    default_value='0.3'),
    ]

    node = Node(
        package='detection_mono',
        executable='detection_mono',
        name='detection_mono',
        output='screen',
        parameters=[{
            'cam_info_topic':       LaunchConfiguration('cam_info_topic'),
            'detections_topic':     LaunchConfiguration('detections_topic'),
            'camera_optical_frame': LaunchConfiguration('camera_optical_frame'),
            'target_frame':         LaunchConfiguration('target_frame'),
            'min_box_height_px':    LaunchConfiguration('min_box_height_px'),
            'max_depth_m':          LaunchConfiguration('max_depth_m'),
            'fallback_depth_m':       LaunchConfiguration('fallback_depth_m'),
            'use_aspect_ratio_yaw':   LaunchConfiguration('use_aspect_ratio_yaw'),
            'enable_tracking':      LaunchConfiguration('enable_tracking'),
            'tracker_max_misses':   LaunchConfiguration('tracker_max_misses'),
            'tracker_gate_m':       LaunchConfiguration('tracker_gate_m'),
            'min_hits_to_publish':  LaunchConfiguration('min_hits_to_publish'),
            'marker_alpha':         LaunchConfiguration('marker_alpha'),
            'marker_lifetime_s':    LaunchConfiguration('marker_lifetime_s'),
        }],
    )

    return LaunchDescription(args + [node])
