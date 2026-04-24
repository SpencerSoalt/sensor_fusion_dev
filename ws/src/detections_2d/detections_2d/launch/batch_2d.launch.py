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
    
    camera1_topic_arg = DeclareLaunchArgument(
        'camera1_topic',
        default_value='/camera1/image_raw',
        description='First camera image topic'
    )
    
    camera2_topic_arg = DeclareLaunchArgument(
        'camera2_topic',
        default_value='/camera6/image_raw',
        description='Second camera image topic'
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
    batch2d_node = Node(
        package='detections_2d',
        executable='batch_2d',
        name='batch_2d',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'camera1_topic': LaunchConfiguration('camera1_topic'),
            'camera2_topic': LaunchConfiguration('camera2_topic'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device': LaunchConfiguration('device'),
        }],
        remappings=[
            # Optional: add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        camera1_topic_arg,
        camera2_topic_arg,
        confidence_threshold_arg,
        device_arg,
        batch2d_node,
    ])
