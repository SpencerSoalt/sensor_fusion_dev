from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Model configuration
        DeclareLaunchArgument(
            'model_path',
            default_value='/ws/yolo12x.pt',
            description='Path to YOLO model file (.pt)'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/camera/color/image_raw',
            description='Input camera topic'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Detection confidence threshold'
        ),
        DeclareLaunchArgument(
            'classes_to_detect',
            default_value="['person', 'car']",
            description='List of class names to detect'
        ),
        
        # TensorRT configuration
        DeclareLaunchArgument(
            'use_tensorrt',
            default_value='true',
            description='Enable TensorRT optimization'
        ),
        DeclareLaunchArgument(
            'tensorrt_precision',
            default_value='fp16',
            description='TensorRT precision: fp32, fp16, or int8'
        ),
        DeclareLaunchArgument(
            'tensorrt_workspace_gb',
            default_value='4',
            description='TensorRT workspace size in GB'
        ),
        DeclareLaunchArgument(
            'input_width',
            default_value='640',
            description='Model input width'
        ),
        DeclareLaunchArgument(
            'input_height',
            default_value='640',
            description='Model input height'
        ),
        DeclareLaunchArgument(
            'batch_size',
            default_value='1',
            description='Inference batch size'
        ),
        DeclareLaunchArgument(
            'dynamic_batch',
            default_value='false',
            description='Enable dynamic batch size'
        ),
        
        # Node
        Node(
            package='detections_2d',
            executable='tensor_rt',
            name='tensor_rt',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'classes_to_detect': LaunchConfiguration('classes_to_detect'),
                'use_tensorrt': LaunchConfiguration('use_tensorrt'),
                'tensorrt_precision': LaunchConfiguration('tensorrt_precision'),
                'tensorrt_workspace_gb': LaunchConfiguration('tensorrt_workspace_gb'),
                'input_width': LaunchConfiguration('input_width'),
                'input_height': LaunchConfiguration('input_height'),
                'batch_size': LaunchConfiguration('batch_size'),
                'dynamic_batch': LaunchConfiguration('dynamic_batch'),
            }],
            remappings=[
                ('/camera/image_raw', LaunchConfiguration('camera_topic')),
            ]
        ),
    ])
