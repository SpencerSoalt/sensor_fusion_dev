#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/ws/yolo12n.pt',
        description='Path to YOLOv12 model weights'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='Detection confidence threshold'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run inference on (cuda or cpu)'
    )
    
    # Create the node
    detection_node = Node(
        package='detections_2d',
        executable='single_2d',
        name='single_2d',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device': LaunchConfiguration('device'),
        }],
        remappings=[
            # Optional: add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        camera_topic_arg,
        confidence_threshold_arg,
        device_arg,
        detection_node,
    ])
