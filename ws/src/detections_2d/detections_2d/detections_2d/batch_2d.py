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


class YOLOv12BatchDetectionNode(Node):
    def __init__(self):
        super().__init__('yolov12_batch_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov12n.pt')
        self.declare_parameter('camera1_topic', '/camera1/image_raw')
        self.declare_parameter('camera2_topic', '/camera2/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('classes_to_detect', ['person', 'car'])
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera1_topic = self.get_parameter('camera1_topic').value
        camera2_topic = self.get_parameter('camera2_topic').value
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
        
        # Image buffers for batch processing
        self.img1 = None
        self.img2 = None
        self.img1_header = None
        self.img2_header = None

        # Best-effort QoS (good for high-rate camera streams)
        image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.sub1 = self.create_subscription(
            Image, camera1_topic, self.camera1_callback, image_qos)
        self.sub2 = self.create_subscription(
            Image, camera2_topic, self.camera2_callback, image_qos)
        
        # Publishers for detections
        self.pub1 = self.create_publisher(
            Detection2DArray, '/camera1/detections', 10)
        self.pub2 = self.create_publisher(
            Detection2DArray, '/camera2/detections', 10)
        
        # Publishers for visualization (optional)
        self.vis_pub1 = self.create_publisher(
            Image, '/camera1/detections_image', 10)
        self.vis_pub2 = self.create_publisher(
            Image, '/camera2/detections_image', 10)
        
        self.get_logger().info('YOLOv12 batch detection node initialized')
    
    def camera1_callback(self, msg):
        self.img1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img1_header = msg.header
        self.process_batch()
    
    def camera2_callback(self, msg):
        self.img2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img2_header = msg.header
        self.process_batch()
    
    def process_batch(self):
        # Only process when both images are available
        if self.img1 is None or self.img2 is None:
            return
        
        # Prepare batch
        images = [self.img1, self.img2]
        
        # Run batch inference with class filtering
        results = self.model(images, conf=self.conf_threshold, classes=self.class_filter, verbose=False)
        
        # Process results for camera 1
        self.publish_detections(results[0], self.pub1, self.img1_header)
        self.publish_visualization(results[0], self.img1, self.vis_pub1, self.img1_header)
        
        # Process results for camera 2
        self.publish_detections(results[1], self.pub2, self.img2_header)
        self.publish_visualization(results[1], self.img2, self.vis_pub2, self.img2_header)
        
        # Clear buffers to avoid reprocessing
        self.img1 = None
        self.img2 = None
    
    def publish_detections(self, result, publisher, header):
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
        
        publisher.publish(detection_array)
    
    def publish_visualization(self, result, img, publisher, header):
        # Draw detections on image
        annotated_img = result.plot()
        
        # Convert back to ROS message
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        publisher.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12BatchDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
