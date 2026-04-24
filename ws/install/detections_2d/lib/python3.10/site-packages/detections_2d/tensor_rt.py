#!/usr/bin/env python3
"""
YOLOv12 Detection Node with TensorRT Optimization for ROS2

This node uses TensorRT to accelerate YOLO inference on NVIDIA GPUs.
On first run, it exports the model to TensorRT format (this takes a few minutes).
Subsequent runs use the cached TensorRT engine for fast inference.
"""

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
from pathlib import Path
import time


class YOLOv12TensorRTNode(Node):
    def __init__(self):
        super().__init__('yolov12_tensorrt_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '/ws/yolov12x.pt')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('classes_to_detect', ['person', 'car', 'truck', 'bus'])
        
        # TensorRT-specific parameters
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('tensorrt_precision', 'fp16')  # 'fp32', 'fp16', or 'int8'
        self.declare_parameter('tensorrt_workspace_gb', 4)  # GPU memory for TRT workspace
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('batch_size', 1)
        self.declare_parameter('dynamic_batch', False)  # Enable for variable batch sizes
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        classes_to_detect = self.get_parameter('classes_to_detect').value
        
        use_tensorrt = self.get_parameter('use_tensorrt').value
        trt_precision = self.get_parameter('tensorrt_precision').value
        workspace_gb = self.get_parameter('tensorrt_workspace_gb').value
        input_width = self.get_parameter('input_width').value
        input_height = self.get_parameter('input_height').value
        batch_size = self.get_parameter('batch_size').value
        dynamic_batch = self.get_parameter('dynamic_batch').value
        
        # Initialize model with TensorRT
        self.model = self._load_model(
            model_path=model_path,
            use_tensorrt=use_tensorrt,
            precision=trt_precision,
            workspace_gb=workspace_gb,
            imgsz=(input_height, input_width),
            batch_size=batch_size,
            dynamic_batch=dynamic_batch
        )
        
        # Map class names to class IDs (COCO dataset)
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
        
        # QoS profile for camera streams
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1, 
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # Subscriber with sensor QoS
        self.sub = self.create_subscription(
            Image, camera_topic, self.camera_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/camera/detections', 10)
        self.vis_pub = self.create_publisher(
            Image, '/camera/detections_image', 10)
        
        # Performance monitoring
        self.inference_times = []
        self.frame_count = 0
        self.perf_timer = self.create_timer(5.0, self.log_performance)
        
        # Fixed color per class (BGR order for OpenCV)
        self._CLASS_COLORS = {
            'person': ( 80,  80, 255),   # red
            'car':    ( 80, 220,  80),   # green
            'bus':    (255, 140,  80),   # blue
            'truck':  (  0, 165, 255),   # orange
        }

        self.get_logger().info(
            f'YOLOv12 TensorRT node initialized, subscribing to {camera_topic}')
    
    def _load_model(self, model_path: str, use_tensorrt: bool, precision: str,
                    workspace_gb: int, imgsz: tuple, batch_size: int,
                    dynamic_batch: bool) -> YOLO:
        """Load YOLO model with optional TensorRT export."""
        
        model_path = Path(model_path)
        
        if use_tensorrt:
            # Determine TensorRT engine path
            precision_suffix = f'_{precision}'
            if dynamic_batch:
                precision_suffix += '_dynamic'
            trt_path = model_path.with_suffix(f'{precision_suffix}.engine')
            
            if trt_path.exists():
                # Load existing TensorRT engine
                self.get_logger().info(f'Loading cached TensorRT engine: {trt_path}')
                model = YOLO(str(trt_path))
            else:
                # Export to TensorRT
                self.get_logger().info(
                    f'Exporting model to TensorRT ({precision})... This may take several minutes.')
                
                # Load PyTorch model first
                model = YOLO(str(model_path))
                
                # Export configuration
                export_args = {
                    'format': 'engine',
                    'imgsz': imgsz,
                    'half': precision in ['fp16'],
                    'int8': precision == 'int8',
                    'device': 0,
                    'workspace': workspace_gb,
                    'verbose': True,
                    'batch': batch_size,
                    'dynamic': dynamic_batch,
                }
                
                # For INT8 quantization, you need calibration data
                if precision == 'int8':
                    self.get_logger().warn(
                        'INT8 quantization requires calibration data for best accuracy. '
                        'Using default calibration.')
                    # Optionally set calibration dataset:
                    # export_args['data'] = 'path/to/calibration/data.yaml'
                
                # Export model
                exported_path = model.export(**export_args)
                self.get_logger().info(f'TensorRT engine exported to: {exported_path}')
                
                # Load the exported engine
                model = YOLO(exported_path)
        else:
            # Use PyTorch model on CUDA
            self.get_logger().info(f'Loading PyTorch model: {model_path}')
            model = YOLO(str(model_path))
            model.to('cuda')
        
        # Warm up the model
        self.get_logger().info('Warming up model...')
        dummy_input = np.zeros((imgsz[0], imgsz[1], 3), dtype=np.uint8)
        for _ in range(10):
            model(dummy_input, verbose=False)
        self.get_logger().info('Model warm-up complete')
        
        return model
    
    def camera_callback(self, msg: Image):
        """Process incoming camera images."""
        start_time = time.perf_counter()
        
        # Convert ROS image to OpenCV
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Run inference
        results = self.model(
            img,
            conf=self.conf_threshold,
            classes=self.class_filter if self.class_filter else None,
            verbose=False
        )
        
        # Track inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Publish results
        self.publish_detections(results[0], msg.header)
        self.publish_visualization(results[0], img, msg.header)
    
    def publish_detections(self, result, header):
        """Publish detection results as Detection2DArray."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.detection_pub.publish(detection_array)
            return
        
        # Process on CPU
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        boxes_cls = boxes.cls.cpu().numpy()
        boxes_conf = boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            detection = Detection2D()
            detection.header = header
            
            # Bounding box (center + size format)
            x1, y1, x2, y2 = boxes_xyxy[i]
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Classification result
            hypothesis = ObjectHypothesisWithPose()
            class_id = int(boxes_cls[i])
            hypothesis.hypothesis.class_id = str(class_id)
            hypothesis.hypothesis.score = float(boxes_conf[i])
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)
    
    def publish_visualization(self, result, img, header):
        """Publish annotated image with per-class coloured bounding boxes."""
        annotated_img = img.copy()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            boxes_cls  = boxes.cls.cpu().numpy()
            boxes_conf = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                class_id   = int(boxes_cls[i])
                conf       = float(boxes_conf[i])
                class_name = self.class_names.get(class_id, str(class_id))

                color = self._CLASS_COLORS.get(class_name, (200, 200, 200))

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                label = f'{class_name} {conf:.2f}'
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated_img,
                              (x1, y1 - th - baseline - 4),
                              (x1 + tw, y1),
                              color, cv2.FILLED)
                cv2.putText(annotated_img, label,
                            (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)

        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        self.vis_pub.publish(vis_msg)
    
    def log_performance(self):
        """Log performance metrics periodically."""
        if not self.inference_times:
            return
        
        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        self.get_logger().info(
            f'Performance: {fps:.1f} FPS | '
            f'Avg: {avg_time:.1f}ms | '
            f'Min: {min_time:.1f}ms | '
            f'Max: {max_time:.1f}ms | '
            f'Frames: {self.frame_count}'
        )
        
        # Reset for next interval
        self.inference_times = []


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12TensorRTNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
