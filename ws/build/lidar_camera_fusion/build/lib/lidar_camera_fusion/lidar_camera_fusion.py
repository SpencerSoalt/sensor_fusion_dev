#!/usr/bin/env python3
"""
lidar_camera_fusion.py

Subscribes:
  /pointcloud        (sensor_msgs/PointCloud2)
  /detections_2d     (vision_msgs/Detection2DArray)

Publishes:
  /fused/cloud_in_boxes  (sensor_msgs/PointCloud2)
      Points that fall inside a 2D detection box, with class_id packed
      into the 'label' field so your DBSCAN node can propagate it downstream.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import cv2
import yaml

from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber


class LidarCameraFusion(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion')

        # Parameters 
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('max_distance', 50.0)
        self.declare_parameter('min_distance', 0.0)
        self.declare_parameter('sync_slop', 0.05)

        calib_path    = self.get_parameter('calibration_file').value
        self.max_dist = self.get_parameter('max_distance').value
        self.min_dist = self.get_parameter('min_distance').value
        slop          = self.get_parameter('sync_slop').value

        #  Load calibration YAML
        self.T_cam_lidar, self.K, self.D, self.img_size = \
            self._load_calibration(calib_path)
        self.get_logger().info(f'Calibration loaded from: {calib_path}')

        # Decompose T for cv2.projectPoints
        self.rvec, _ = cv2.Rodrigues(self.T_cam_lidar[:3, :3])
        self.tvec     = self.T_cam_lidar[:3, 3]

        # Subscribers 
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.sub_pc   = Subscriber(self, PointCloud2,      '/velodyne_points',    10) #qos_profile=qos
        self.sub_dets = Subscriber(self, Detection2DArray, '/camera/detections', 10) #qos_profile=qos

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_pc, self.sub_dets],
            queue_size=10,
            slop=slop,
        )
        self.sync.registerCallback(self.fusion_callback)

        #  Publisher 
        self.pub_masked_cloud = self.create_publisher(
            PointCloud2, '/fused/cloud_in_boxes', qos
        )

        self.get_logger().info('Fusion node ready.')


    def _load_calibration(self, path: str):
            with open(path, 'r') as f:
                c = yaml.safe_load(f)

            # Accept either a flat 16-element 4x4 matrix (preferred)
            ext = c['extrinsic']
            T = np.array(ext['matrix'], dtype=np.float64).reshape(4, 4)
   

            # Inverse T_cam_lidar (LiDAR -> camera) for projection.
            T = np.linalg.inv(T)

            K = np.array([
                [c['intrinsic']['fx'],               0., c['intrinsic']['cx']],
                [              0., c['intrinsic']['fy'], c['intrinsic']['cy']],
                [              0.,               0.,              1.         ],
            ], dtype=np.float64)

            D        = np.array(c['intrinsic'].get('distortion', [0., 0., 0., 0., 0.]))
            img_size = (c['intrinsic']['width'], c['intrinsic']['height'])

            return T, K, D, img_size

    def _project_points(self, xyz: np.ndarray):
        """
        Project Nx3 LiDAR points into the image plane.

        Returns
        -------
        uv         : Nx2  pixel coordinates (float)
        valid_mask : N    bool — in front of camera, within image bounds,
                               and within [min_dist, max_dist]
        """
        # Depth in camera frame (needed for distance filtering)
        pts_cam = (self.T_cam_lidar[:3, :3] @ xyz.T).T + self.T_cam_lidar[:3, 3]
        Z_cam   = pts_cam[:, 2]

        # Project to image plane
        uv, _ = cv2.projectPoints(
            xyz.astype(np.float64),
            self.rvec,
            self.tvec,
            self.K,
            self.D,
        )
        uv = uv.reshape(-1, 2)

        W, H       = self.img_size
        u, v       = uv[:, 0], uv[:, 1]
        in_bounds  = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dist_ok    = (Z_cam > self.min_dist) & (Z_cam < self.max_dist)
        valid_mask = in_bounds & dist_ok

        return uv, valid_mask

    def fusion_callback(self, pc_msg: PointCloud2, det_msg: Detection2DArray):

        # Read point cloud 
        cloud_raw = pc2.read_points(
            pc_msg, field_names=('x', 'y', 'z'), skip_nans=True
        )
        cloud_arr = np.array(list(cloud_raw))
        if cloud_arr.size == 0:
            return
        xyz = np.column_stack([cloud_arr['x'], cloud_arr['y'], cloud_arr['z']]).astype(np.float32)  # N x 3

        # Parse 2D detections
        boxes = []  # (x1, y1, x2, y2, class_id)
        for det in det_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w  = det.bbox.size_x
            h  = det.bbox.size_y

            class_id = 0
            if det.results:
                cid_str  = det.results[0].hypothesis.class_id
                class_id = int(cid_str) if cid_str.isdigit() else 0

            boxes.append((cx - w / 2., cy - h / 2.,
                          cx + w / 2., cy + h / 2.,
                          class_id))

        if not boxes:
            return

        # Project LiDAR to image plane 
        uv, valid_mask = self._project_points(xyz)

        valid_xyz = xyz[valid_mask]       # M x 3  (points on screen)
        valid_uv  = uv[valid_mask]        # M x 2

        if len(valid_xyz) == 0:
            return

        # Assign each projected point to a detection box 
        u_arr     = valid_uv[:, 0]
        v_arr     = valid_uv[:, 1]
        label_ids = np.full(len(valid_xyz), -1, dtype=np.int32)

        for (x1, y1, x2, y2, class_id) in boxes:
            inside            = (u_arr >= x1) & (u_arr <= x2) & \
                                (v_arr >= y1) & (v_arr <= y2)
            label_ids[inside] = class_id   # later boxes overwrite on overlap

        # Keep only points inside at least one box
        in_box       = label_ids >= 0
        pts_in_boxes = valid_xyz[in_box]
        pts_labels   = label_ids[in_box]

        self.get_logger().debug(
            f'Projected {len(valid_xyz)} pts | {in_box.sum()} inside boxes'
        )

        # Publish labelled masked cloud 
        self._publish_labeled_cloud(pts_in_boxes, pts_labels, pc_msg.header)

    def _publish_labeled_cloud(self, pts: np.ndarray,
                                      labels: np.ndarray,
                                      header: Header):
        """
        PointCloud2 with fields x, y, z, label (float32 cast of class_id).
        Your DBSCAN node reads 'label' and applies majority-vote per cluster.
        """
        if len(pts) == 0:
            return

        fields = [
            PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        data      = np.hstack([pts, labels.reshape(-1, 1).astype(np.float32)])
        cloud_msg = pc2.create_cloud(header, fields, data)
        self.pub_masked_cloud.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()