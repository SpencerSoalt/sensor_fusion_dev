#!/usr/bin/env python3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
    Pose2D,
)
from geometry_msgs.msg import PoseWithCovariance

# Ultralytics YOLO (YOLO12 / YOLOv12)
from ultralytics import YOLO

# Time sync
from message_filters import Subscriber, ApproximateTimeSynchronizer


COCO_DEFAULT_LABELS = {
    0: "person",
    2: "car",
}


class Yolo12TwoCamDetector(Node):
    """
    Subscribes to two Image topics, time-syncs them (approx), runs ONE batched YOLO inference,
    publishes Detection2DArray per camera.

    Intended for rosbag-based testing (set use_sim_time:=true when playing bags with --clock).
    """

    def __init__(self):
        super().__init__("yolo12_2d_detector")

        # ---------- Parameters ----------
        # Your real topics (commented for reference):
        # Left:  /camera4/image_raw/compressed  (NOTE: that's CompressedImage, not Image)
        # Right: /camera1/image_raw/compressed

        # Current (Image) defaults:
        self.declare_parameter("left_image_topic", "/front_left/image_raw")
        self.declare_parameter("right_image_topic", "/front_right/image_raw")

        self.declare_parameter("left_detections_topic", "/front_left/detections2d")
        self.declare_parameter("right_detections_topic", "/front_right/detections2d")

        self.declare_parameter("publish_debug_images", True)
        self.declare_parameter("left_debug_image_topic", "/front_left/detections_image")
        self.declare_parameter("right_debug_image_topic", "/front_right/detections_image")

        # YOLO params
        self.declare_parameter("model", "yolo12n.pt")  # COCO-pretrained by default
        self.declare_parameter("device", "")  # e.g. "cuda:0" or "cpu"; empty lets ultralytics choose
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("classes", [0, 2])  # COCO: person=0, car=2
        self.declare_parameter("use_class_names", True)

        # Sync params
        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("sync_slop_sec", 0.05)  # 50ms tolerance

        # Behavior
        self.declare_parameter("skip_if_busy", True)  # drop synced pairs if inference still running

        # ---------- Read parameters ----------
        self.left_image_topic = self.get_parameter("left_image_topic").value
        self.right_image_topic = self.get_parameter("right_image_topic").value

        self.left_det_topic = self.get_parameter("left_detections_topic").value
        self.right_det_topic = self.get_parameter("right_detections_topic").value

        self.publish_debug = bool(self.get_parameter("publish_debug_images").value)
        self.left_dbg_topic = self.get_parameter("left_debug_image_topic").value
        self.right_dbg_topic = self.get_parameter("right_debug_image_topic").value

        self.model_name = self.get_parameter("model").value
        self.device = self.get_parameter("device").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.class_filter = list(self.get_parameter("classes").value)
        self.use_class_names = bool(self.get_parameter("use_class_names").value)

        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop_sec").value)

        self.skip_if_busy = bool(self.get_parameter("skip_if_busy").value)

        self.get_logger().info(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)

        # Ultralytics model access is not guaranteed thread-safe. Use a lock.
        self.model_lock = threading.Lock()

        # Busy flag (we keep two for backward-compat with your code style)
        self.left_busy = threading.Event()
        self.right_busy = threading.Event()

        # Executor for running inference without blocking ROS callback threads.
        # Batched inference runs as one job, so 1 worker is enough.
        self.pool = ThreadPoolExecutor(max_workers=1)

        self.bridge = CvBridge()

        # QoS suitable for camera streams
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.left_pub = self.create_publisher(Detection2DArray, self.left_det_topic, 10)
        self.right_pub = self.create_publisher(Detection2DArray, self.right_det_topic, 10)

        self.left_dbg_pub = None
        self.right_dbg_pub = None
        if self.publish_debug:
            self.left_dbg_pub = self.create_publisher(Image, self.left_dbg_topic, 10)
            self.right_dbg_pub = self.create_publisher(Image, self.right_dbg_topic, 10)

        # ---------- Subscribers + Approximate Time Sync ----------
        self.left_sub = Subscriber(self, Image, self.left_image_topic, qos_profile=qos)
        self.right_sub = Subscriber(self, Image, self.right_image_topic, qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_cb)

        self.get_logger().info("YOLO12 2D detector node started (batched inference enabled).")
        self.get_logger().info(f"Left  image: {self.left_image_topic} -> {self.left_det_topic}")
        self.get_logger().info(f"Right image: {self.right_image_topic} -> {self.right_det_topic}")
        self.get_logger().info(f"Filtering classes: {self.class_filter} (COCO person=0, car=2)")
        self.get_logger().info(f"Sync: queue_size={self.sync_queue_size}, slop={self.sync_slop}s")

    def _synced_cb(self, left_msg: Image, right_msg: Image):
        # Drop pairs if still processing previous pair
        if self.skip_if_busy and (self.left_busy.is_set() or self.right_busy.is_set()):
            return

        self.left_busy.set()
        self.right_busy.set()

        # Do all heavy work off-thread
        self.pool.submit(self._run_and_publish_batched, left_msg, right_msg)

    def _run_and_publish_batched(self, left_msg: Image, right_msg: Image):
        try:
            # Convert both images
            try:
                left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().error(f"Left cv_bridge conversion failed: {e}")
                return

            try:
                right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().error(f"Right cv_bridge conversion failed: {e}")
                return

            # One batched YOLO call
            with self.model_lock:
                results = self.model.predict(
                    source=[left_cv, right_cv],
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    classes=self.class_filter if len(self.class_filter) > 0 else None,
                    device=self.device if self.device else None,
                    verbose=False,
                )

            # results[0] corresponds to left_cv, results[1] to right_cv
            left_det, left_dbg = self._result_to_detection2darray(left_msg, left_cv, results[0])
            right_det, right_dbg = self._result_to_detection2darray(right_msg, right_cv, results[1])

            self.left_pub.publish(left_det)
            self.right_pub.publish(right_det)

            if self.publish_debug and self.left_dbg_pub is not None and left_dbg is not None:
                self.left_dbg_pub.publish(left_dbg)
            if self.publish_debug and self.right_dbg_pub is not None and right_dbg is not None:
                self.right_dbg_pub.publish(right_dbg)

        except Exception as e:
            self.get_logger().error(f"batched inference/publish error: {e}")
        finally:
            self.left_busy.clear()
            self.right_busy.clear()

    def _result_to_detection2darray(
        self, img_msg: Image, cv_img_bgr: np.ndarray, r0
    ) -> Tuple[Detection2DArray, Optional[Image]]:
        det_msg = Detection2DArray()
        det_msg.header = img_msg.header  # timestamp + frame_id from the image

        # Prepare debug overlay if enabled
        dbg_msg = None
        dbg_img = cv_img_bgr.copy() if self.publish_debug else None

        # Ultralytics boxes: xyxy, cls, conf
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            cls = r0.boxes.cls.cpu().numpy().astype(int)
            conf = r0.boxes.conf.cpu().numpy()

            for i, (box, c, s) in enumerate(zip(xyxy, cls, conf)):
                x1, y1, x2, y2 = box.tolist()

                d = Detection2D()
                d.header = img_msg.header

                # bbox as center + size (pixels)
                bbox = BoundingBox2D()
                bbox.center = Pose2D()
                bbox.center.position.x = float((x1 + x2) / 2.0)
                bbox.center.position.y = float((y1 + y2) / 2.0)
                bbox.center.theta = 0.0
                bbox.size_x = float(max(0.0, x2 - x1))
                bbox.size_y = float(max(0.0, y2 - y1))
                d.bbox = bbox

                # hypothesis
                hyp = ObjectHypothesisWithPose()
                class_id = (
                    COCO_DEFAULT_LABELS.get(int(c), str(int(c)))
                    if self.use_class_names
                    else str(int(c))
                )
                hyp.hypothesis.class_id = str(class_id)
                hyp.hypothesis.score = float(s)
                hyp.pose = PoseWithCovariance()  # left default/zero (2D output)
                d.results = [hyp]

                # optional stable-ish ID within this message
                d.id = str(i)

                det_msg.detections.append(d)

                # debug draw
                if dbg_img is not None:
                    import cv2

                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(dbg_img, p1, p2, (0, 255, 0), 2)
                    label = f"{class_id}:{s:.2f}"
                    cv2.putText(
                        dbg_img,
                        label,
                        (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        if dbg_img is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg_img, encoding="bgr8")
            dbg_msg.header = img_msg.header

        return det_msg, dbg_msg


def main():
    rclpy.init()
    node = Yolo12TwoCamDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
