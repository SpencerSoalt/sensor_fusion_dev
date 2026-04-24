#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class YOLOv12DetectionNode(Node):
    def __init__(self):
        super().__init__('yolov12_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov12x.pt')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('classes_to_detect', ['person', 'car'])
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        classes_to_detect = self.get_parameter('classes_to_detect').value
        
        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLOv12 model from {model_path}')
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Map class names to class IDs
        # COCO dataset: person=0, car=2
        self.class_names = self.model.names
        self.class_filter = []
        for class_name in classes_to_detect:
            for class_id, name in self.class_names.items():
                if name.lower() == class_name.lower():
                    self.class_filter.append(class_id)
                    break
        
        self.get_logger().info(f'Filtering for classes: {classes_to_detect}')
        self.get_logger().info(f'Class IDs: {self.class_filter}')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Best-effort QoS (good for high-rate camera streams)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscriber
        self.sub = self.create_subscription(
            Image, camera_topic, self.camera_callback, 10)
        
        # Publishers for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/camera/detections', 10)
        
        # Publisher for visualization (optional)
        self.vis_pub = self.create_publisher(
            Image, '/camera/detections_image', 10)
        
        self.get_logger().info(f'YOLOv12 detection node initialized, subscribing to {camera_topic}')
    
    def camera_callback(self, msg):
        # Convert ROS image to OpenCV
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run inference with class filtering
        results = self.model(img, conf=self.conf_threshold, classes=self.class_filter, verbose=False)
        
        # Process and publish results
        self.publish_detections(results[0], msg.header)
        self.publish_visualization(results[0], img, msg.header)
    
    def publish_detections(self, result, header):
        detection_array = Detection2DArray()
        detection_array.header = header
        
        boxes = result.boxes
        for i in range(len(boxes)):
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(boxes.cls[i].item()))
            hypothesis.hypothesis.score = float(boxes.conf[i].item())
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)
    
    def publish_visualization(self, result, img, header):
        # Draw detections on image
        annotated_img = result.plot()
        
        # Convert back to ROS message
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        self.vis_pub.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12DetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
