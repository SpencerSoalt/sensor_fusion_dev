#!/usr/bin/env python3
"""
LiDAR + 2D Detection fusion node with 3D bounding box estimation.

Features:
  1. ApproximateTimeSynchronizer for LiDAR/detection temporal alignment
  2. Center-crop bbox (configurable) to reduce depth contamination
  3. Class-based 3D size priors with optional L-shape refinement from
     LiDAR points when enough points are available
  4. Per-object Kalman tracking (constant-velocity) with Hungarian
     assignment for stable IDs and smooth positions
  5. Publishes CUBE markers (oriented 3D boxes) instead of fixed spheres
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped

from vision_msgs.msg import Detection2DArray

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point

from sensor_msgs_py import point_cloud2

from message_filters import Subscriber, ApproximateTimeSynchronizer

from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Class-based 3D size priors: (length, width, height) in meters
# These are typical values — tune to your domain.
# ---------------------------------------------------------------------------
SIZE_PRIORS = {
    # COCO / YOLO-style labels
    "car":          (4.5, 1.8, 1.5),
    "truck":        (6.5, 2.5, 2.5),
    "bus":          (10.0, 2.5, 3.2),
    "motorcycle":   (2.2, 0.8, 1.4),
    "bicycle":      (1.8, 0.6, 1.2),
    "person":       (0.5, 0.5, 1.7),
    "pedestrian":   (0.5, 0.5, 1.7),
    "dog":          (0.8, 0.4, 0.6),
    "cat":          (0.5, 0.3, 0.3),
    "stop sign":    (0.6, 0.1, 0.7),
    "fire hydrant": (0.3, 0.3, 0.5),
    "bench":        (1.5, 0.5, 0.8),
    "traffic light": (0.3, 0.3, 0.9),
}

DEFAULT_SIZE = (1.0, 1.0, 1.0)

# COCO class ID → name (subset relevant to outdoor/driving scenes)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    15: "cat", 16: "dog",
}

# Classes where L-shape fitting makes sense (roughly box-shaped vehicles)
LSHAPE_CLASSES = {"car", "truck", "bus"}

# Minimum LiDAR points needed to attempt L-shape fitting
LSHAPE_MIN_POINTS = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy)],
        [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx)],
        [      2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float32)


def yaw_to_quaternion(yaw: float):
    """Yaw angle (radians, around Z-up) -> quaternion (x,y,z,w)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ---------------------------------------------------------------------------
# L-shape fitting
# ---------------------------------------------------------------------------

def fit_lshape_yaw(pts_xy: np.ndarray) -> tuple[float, float, float]:
    """
    Simple L-shape / minimum-area rectangle fitting on 2D points.

    Searches discrete yaw angles, rotates points, computes axis-aligned
    bounding box area, and picks the yaw that minimizes it.

    Args:
        pts_xy: (N, 2) points in the ground plane (X-forward, Y-left)

    Returns:
        (yaw, fitted_length, fitted_width)
    """
    best_yaw = 0.0
    best_area = float("inf")
    best_dims = (1.0, 1.0)

    for deg in range(0, 180, 2):
        theta = math.radians(deg)
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, s], [-s, c]], dtype=np.float32)
        rotated = pts_xy @ rot.T

        mn = rotated.min(axis=0)
        mx = rotated.max(axis=0)
        dx = mx[0] - mn[0]
        dy = mx[1] - mn[1]
        area = dx * dy

        if area < best_area:
            best_area = area
            best_yaw = theta
            best_dims = (float(dx), float(dy))

    length = max(best_dims)
    width = min(best_dims)

    # Ensure yaw aligns with the longer dimension
    c, s = math.cos(best_yaw), math.sin(best_yaw)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)
    rotated = pts_xy @ rot.T
    mn = rotated.min(axis=0)
    mx = rotated.max(axis=0)
    dx = mx[0] - mn[0]

    if dx < (mx[1] - mn[1]):
        best_yaw += math.pi / 2.0

    # Normalize to [-pi, pi]
    best_yaw = math.atan2(math.sin(best_yaw), math.cos(best_yaw))

    return best_yaw, length, width


# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------

@dataclass
class Detection3D:
    """Single 3D detection before tracking."""
    position: np.ndarray   # (3,) in target frame
    size: tuple            # (length, width, height)
    yaw: float             # radians around Z
    label: str


# ---------------------------------------------------------------------------
# Constant-velocity Kalman tracker
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Single tracked object with a 6-state constant-velocity Kalman filter.

    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """
    track_id: int
    x: np.ndarray            # state (6,)
    P: np.ndarray            # covariance (6,6)
    age: int = 0
    hits: int = 1
    misses: int = 0
    label: str = ""
    yaw: float = 0.0
    size: tuple = (1.0, 1.0, 1.0)

    _q_pos: float = 0.5
    _q_vel: float = 2.0
    _r: float = 0.3

    def predict(self, dt: float):
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = np.zeros((6, 6), dtype=np.float32)
        for i in range(3):
            Q[i, i] = self._q_pos * dt ** 2
            Q[i + 3, i + 3] = self._q_vel * dt ** 2

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, det: Detection3D):
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3, dtype=np.float32) * self._r

        z = det.position
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float32) - K @ H) @ self.P

        self.hits += 1
        self.misses = 0
        self.label = det.label

        # Exponential moving average for yaw and size to smooth jitter
        alpha = 0.4
        # Unwrap yaw difference
        dyaw = math.atan2(
            math.sin(det.yaw - self.yaw),
            math.cos(det.yaw - self.yaw),
        )
        self.yaw += alpha * dyaw
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        self.size = tuple(
            (1 - alpha) * s_old + alpha * s_new
            for s_old, s_new in zip(self.size, det.size)
        )

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]


class MultiObjectTracker:
    """Multi-object tracker with Hungarian assignment."""

    def __init__(self, max_misses: int = 5, gate_distance: float = 3.0):
        self._next_id = 0
        self.tracks: list[Track] = []
        self.max_misses = max_misses
        self.gate_distance = gate_distance
        self._last_time: float | None = None

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def step(self, detections: list[Detection3D], stamp) -> list[Track]:
        t = self._stamp_to_sec(stamp)
        dt = 0.033 if self._last_time is None else max(t - self._last_time, 1e-4)
        self._last_time = t

        for trk in self.tracks:
            trk.predict(dt)
            trk.age += 1

        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = np.linalg.norm(trk.position - det.position)

            row_idx, col_idx = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < self.gate_distance:
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            for i in range(n_tracks):
                if i not in matched_tracks:
                    self.tracks[i].misses += 1

            for j in range(n_dets):
                if j not in matched_dets:
                    self._create_track(detections[j])
        elif n_dets > 0:
            for det in detections:
                self._create_track(det)
        else:
            for trk in self.tracks:
                trk.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def _create_track(self, det: Detection3D):
        x = np.zeros(6, dtype=np.float32)
        x[:3] = det.position
        P = np.eye(6, dtype=np.float32)
        P[0, 0] = P[1, 1] = P[2, 2] = 1.0
        P[3, 3] = P[4, 4] = P[5, 5] = 10.0
        trk = Track(
            track_id=self._next_id, x=x, P=P,
            label=det.label, yaw=det.yaw, size=det.size,
        )
        self._next_id += 1
        self.tracks.append(trk)


# ---------------------------------------------------------------------------
# Main ROS2 Node
# ---------------------------------------------------------------------------

class Detection2_5D(Node):
    def __init__(self):
        super().__init__('lidar_bbox_fusion')

        # --- Parameters: topics & frames ---
        self.declare_parameter('cloud_topic', '/velodyne_points')
        self.declare_parameter('cam_info_topic', '/camera1/camera_info')
        self.declare_parameter('detections_topic', '/camera1/detections')

        self.declare_parameter('lidar_frame', 'velodyne_link')
        self.declare_parameter('camera_optical_frame', 'camera_left_optical_frame')
        self.declare_parameter('target_frame', 'base_link')

        # --- Parameters: point cloud processing ---
        self.declare_parameter('min_points_in_box', 10)
        self.declare_parameter('max_points_used', 60000)
        self.declare_parameter('stride', 2)
        self.declare_parameter('max_range_m', 30.0)

        # --- Parameters: depth / crop ---
        self.declare_parameter('publish_when_no_depth', True)
        self.declare_parameter('fallback_depth_m', 15.0)
        self.declare_parameter('bbox_crop_ratio', 0.5)

        # --- Parameters: 3D box estimation ---
        self.declare_parameter('use_lshape_fitting', True)
        self.declare_parameter('lshape_min_points', 25)

        # --- Parameters: tracker ---
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('tracker_max_misses', 5)
        self.declare_parameter('tracker_gate_m', 3.0)
        self.declare_parameter('min_hits_to_publish', 2)

        # --- Parameters: time sync ---
        self.declare_parameter('sync_slop_s', 0.1)
        self.declare_parameter('sync_queue_size', 15)

        # --- Parameters: visualization ---
        self.declare_parameter('marker_alpha', 0.6)
        self.declare_parameter('marker_lifetime_s', 0.3)

        # --- State ---
        self.have_cam_info = False
        self.K = self.D = None
        self.fx = self.fy = self.cx = self.cy = None

        self.tracker = MultiObjectTracker(
            max_misses=int(self.get_parameter('tracker_max_misses').value),
            gate_distance=float(self.get_parameter('tracker_gate_m').value),
        )

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscribers ---
        cloud_topic = self.get_parameter('cloud_topic').value
        cam_info_topic = self.get_parameter('cam_info_topic').value
        det_topic = self.get_parameter('detections_topic').value

        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.on_info, 10
        )

        slop = float(self.get_parameter('sync_slop_s').value)
        queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sub_cloud = Subscriber(self, PointCloud2, cloud_topic)
        self.sub_det = Subscriber(self, Detection2DArray, det_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_cloud, self.sub_det],
            queue_size=queue_size,
            slop=slop,
        )
        self.sync.registerCallback(self.on_synced)

        # --- Publishers ---
        self.pub_markers = self.create_publisher(
            MarkerArray, '/detection_boxes_3d', 10
        )

        self.get_logger().info("LiDAR + 2D detection → 3D box fusion (synced, tracked)")
        self.get_logger().info(f"  Cloud:      {cloud_topic}")
        self.get_logger().info(f"  CameraInfo: {cam_info_topic}")
        self.get_logger().info(f"  Detections: {det_topic}")
        self.get_logger().info(f"  Sync slop:  {slop:.3f}s")
        self.get_logger().info("  Output:     /detection_boxes_3d")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float32).reshape(-1)
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.have_cam_info = True

    def on_synced(self, cloud_msg: PointCloud2, det_msg: Detection2DArray):
        if not self.have_cam_info:
            return

        lidar_frame = self.get_parameter('lidar_frame').value
        cam_frame = self.get_parameter('camera_optical_frame').value
        target_frame = self.get_parameter('target_frame').value

        min_pts = int(self.get_parameter('min_points_in_box').value)
        stride = int(self.get_parameter('stride').value)
        max_pts_used = int(self.get_parameter('max_points_used').value)
        max_range = float(self.get_parameter('max_range_m').value)
        publish_when_no_depth = bool(self.get_parameter('publish_when_no_depth').value)
        fallback_depth = float(self.get_parameter('fallback_depth_m').value)
        crop_ratio = float(self.get_parameter('bbox_crop_ratio').value)
        use_lshape = bool(self.get_parameter('use_lshape_fitting').value)
        lshape_min = int(self.get_parameter('lshape_min_points').value)

        # --- TF lookups ---
        try:
            R_cv, t_cv, _ = self._get_tf_as_RT(cam_frame, lidar_frame)
        except TransformException as ex:
            self.get_logger().warn(f"TF cam<-lidar failed: {ex}")
            return

        try:
            R_bt, t_bt, tf_base_cam = self._get_tf_as_RT(target_frame, cam_frame)
        except TransformException as ex:
            self.get_logger().warn(f"TF base<-cam failed: {ex}")
            return

        # Also need base_link <- lidar for L-shape fitting in target frame
        try:
            R_bl, t_bl, _ = self._get_tf_as_RT(target_frame, lidar_frame)
        except TransformException as ex:
            self.get_logger().warn(f"TF base<-lidar failed: {ex}")
            return

        # --- Read and subsample LiDAR cloud ---
        pts = []
        step_count = 0
        for p in point_cloud2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            if stride > 1 and (step_count % stride) != 0:
                step_count += 1
                continue
            step_count += 1

            x, y, z = float(p[0]), float(p[1]), float(p[2])
            r = math.sqrt(x * x + y * y + z * z)
            if r < 0.5 or r > max_range:
                continue
            pts.append((x, y, z))
            if len(pts) >= max_pts_used:
                break

        if len(pts) == 0:
            return

        pts_lidar = np.asarray(pts, dtype=np.float32)

        # Transform to camera optical frame
        pts_cam = (R_cv @ pts_lidar.T).T + t_cv

        # Keep only in front of camera
        in_front = pts_cam[:, 2] > 0.2
        pts_cam = pts_cam[in_front]
        pts_lidar_front = pts_lidar[in_front]   # matching lidar-frame points
        if pts_cam.shape[0] < 20:
            return

        # Project into raw image pixels
        uv = self._project_points_raw(pts_cam)

        # --- Per-detection: depth + 3D box estimation ---
        stamp = det_msg.header.stamp
        detections_3d: list[Detection3D] = []

        for det in det_msg.detections:
            uc = float(det.bbox.center.position.x)
            vc = float(det.bbox.center.position.y)
            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)

            # Center-cropped bbox for depth estimation
            cw, ch = w * crop_ratio, h * crop_ratio
            umin_crop = uc - cw / 2.0
            umax_crop = uc + cw / 2.0
            vmin_crop = vc - ch / 2.0
            vmax_crop = vc + ch / 2.0

            inside = (
                (uv[:, 0] >= umin_crop) & (uv[:, 0] <= umax_crop) &
                (uv[:, 1] >= vmin_crop) & (uv[:, 1] <= vmax_crop)
            )

            pts_in_cam = pts_cam[inside]
            pts_in_lidar = pts_lidar_front[inside]

            if pts_in_cam.shape[0] < min_pts:
                if not publish_when_no_depth:
                    continue
                Z = fallback_depth
            else:
                Z = float(np.median(pts_in_cam[:, 2]))

            # Back-project bbox center to 3D in camera frame
            X = (uc - self.cx) * Z / self.fx
            Y = (vc - self.cy) * Z / self.fy

            p_cam = PointStamped()
            p_cam.header.frame_id = cam_frame
            p_cam.header.stamp = stamp
            p_cam.point.x = float(X)
            p_cam.point.y = float(Y)
            p_cam.point.z = float(Z)

            p_base = do_transform_point(p_cam, tf_base_cam)
            pos = np.array([
                p_base.point.x, p_base.point.y, p_base.point.z
            ], dtype=np.float32)

            # Extract label — tensor_rt.py publishes numeric COCO IDs as strings
            label = ""
            if det.results:
                raw = det.results[0].hypothesis.class_id
                try:
                    label = COCO_NAMES.get(int(raw), raw)
                except (ValueError, TypeError):
                    label = raw
            label_lower = label.lower()

            # --- 3D size estimation ---
            size_prior = SIZE_PRIORS.get(label_lower, DEFAULT_SIZE)
            yaw = 0.0

            if (
                use_lshape
                and label_lower in LSHAPE_CLASSES
                and pts_in_lidar.shape[0] >= lshape_min
            ):
                # Transform frustum points into target frame for fitting
                pts_base = (R_bl @ pts_in_lidar.T).T + t_bl
                pts_xy = pts_base[:, :2]  # ground plane

                fitted_yaw, fitted_len, fitted_width = fit_lshape_yaw(pts_xy)

                # Sanity check: reject wild fits (> 2x or < 0.3x prior)
                prior_len, prior_wid, prior_h = size_prior
                if (
                    0.3 * prior_len < fitted_len < 2.0 * prior_len
                    and 0.3 * prior_wid < fitted_width < 2.0 * prior_wid
                ):
                    size = (fitted_len, fitted_width, prior_h)
                    yaw = fitted_yaw
                else:
                    size = size_prior
            else:
                size = size_prior

            detections_3d.append(Detection3D(
                position=pos, size=size, yaw=yaw, label=label,
            ))

        # --- Tracker update (or pass-through if tracking disabled) ---
        if bool(self.get_parameter('enable_tracking').value):
            active_tracks = self.tracker.step(detections_3d, stamp)
        else:
            active_tracks = [
                Track(
                    track_id=i, x=np.concatenate([d.position, np.zeros(3, dtype=np.float32)]),
                    P=np.eye(6, dtype=np.float32),
                    label=d.label, yaw=d.yaw, size=d.size, hits=999,
                )
                for i, d in enumerate(detections_3d)
            ]

        # --- Publish 3D box markers ---
        self._publish_markers(active_tracks, stamp)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_markers(self, tracks: list[Track], stamp):
        target_frame = self.get_parameter('target_frame').value
        alpha = float(self.get_parameter('marker_alpha').value)
        lifetime_s = float(self.get_parameter('marker_lifetime_s').value)
        min_hits = int(self.get_parameter('min_hits_to_publish').value)

        marr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        marr.markers.append(clear)

        # Color palette per class
        colors = {
            "car":        (0.2, 0.8, 1.0),
            "truck":      (1.0, 0.6, 0.1),
            "bus":        (1.0, 0.8, 0.0),
            "motorcycle": (0.9, 0.3, 0.9),
            "bicycle":    (0.3, 1.0, 0.3),
            "person":     (1.0, 0.2, 0.2),
            "pedestrian": (1.0, 0.2, 0.2),
        }
        default_color = (0.5, 0.5, 0.5)

        for trk in tracks:
            if trk.hits < min_hits:
                continue

            # --- 3D box marker ---
            m = Marker()
            m.header.frame_id = target_frame
            m.header.stamp = stamp
            m.ns = "tracked_boxes"
            m.id = trk.track_id
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(trk.position[0])
            m.pose.position.y = float(trk.position[1])
            m.pose.position.z = float(trk.position[2])

            qx, qy, qz, qw = yaw_to_quaternion(trk.yaw)
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw

            length, width, height = trk.size
            m.scale.x = length
            m.scale.y = width
            m.scale.z = height

            r, g, b = colors.get(trk.label.lower(), default_color)
            m.color.a = alpha
            m.color.r = r
            m.color.g = g
            m.color.b = b

            m.lifetime = rclpy.duration.Duration(seconds=lifetime_s).to_msg()
            marr.markers.append(m)

            # --- Label text marker ---
            txt = Marker()
            txt.header.frame_id = target_frame
            txt.header.stamp = stamp
            txt.ns = "tracked_labels"
            txt.id = trk.track_id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD

            txt.pose.position.x = float(trk.position[0])
            txt.pose.position.y = float(trk.position[1])
            txt.pose.position.z = float(trk.position[2]) + height / 2.0 + 0.3

            txt.scale.z = 0.4  # text height
            txt.color.a = 1.0
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0

            txt.text = f"{trk.label} #{trk.track_id}"
            txt.lifetime = rclpy.duration.Duration(seconds=lifetime_s).to_msg()
            marr.markers.append(txt)

        self.pub_markers.publish(marr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tf_as_RT(self, target_frame: str, source_frame: str):
        tf = self.tf_buffer.lookup_transform(
            target_frame, source_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.05),
        )
        tr = tf.transform.translation
        rq = tf.transform.rotation
        R = quat_to_rot(rq.x, rq.y, rq.z, rq.w)
        t = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        return R, t, tf

    def _project_points_raw(self, pts_cam_xyz: np.ndarray):
        obj = pts_cam_xyz.reshape(-1, 1, 3).astype(np.float32)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        uv, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
        return uv.reshape(-1, 2).astype(np.float32)


def main():
    rclpy.init()
    node = Detection2_5D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

