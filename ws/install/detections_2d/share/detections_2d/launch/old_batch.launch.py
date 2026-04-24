from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="detections_2d",
            executable="old_batch",
            name="old_batch",
            output="screen",
            parameters=[{
                "use_sim_time": True,  # for rosbag playback with --clock
                
                # subscribe to these topics
                "left_image_topic": "/front_left/image_raw",
                "right_image_topic": "/front_right/image_raw",
                
                # publish to these topics
                "left_detections_topic": "/front_left/detections2d",
                "right_detections_topic": "/front_right/detections2d",

                "publish_debug_images": True,
                "model": "yolo12n.pt",
                "classes": [0, 2],  # person, car
                "conf": 0.25,
                "iou": 0.45,
                "imgsz": 640,
                "skip_if_busy": True,

                # batch/sync-specific params added by the batch node
                "sync_queue_size": 5,
                "sync_slop_sec": 0.05,
            }],
        )
    ])
