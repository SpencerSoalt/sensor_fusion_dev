#!/usr/bin/env python3
"""
Camera-only 2D detection → pseudo-3D node using size-prior back-projection.

Depth is estimated per detection as:
    Z = fy * real_height / pixel_height

where real_height comes from class-based size priors.  No LiDAR required.
The rest of the pipeline (Kalman tracking, 3D box markers) is identical to
the LiDAR-fusion node so outputs are drop-in compatible.
"""

import math
import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point

from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Class-based 3D size priors: (length, width, height) in metres
# ---------------------------------------------------------------------------
SIZE_PRIORS = {
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

# Classes where aspect-ratio yaw estimation is meaningful (elongated vehicles)
YAW_ESTIMATE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    15: "cat", 16: "dog",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1 - 2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)


def yaw_to_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def estimate_yaw_from_aspect(
    bbox_w_px: float,
    bbox_h_px: float,
    prior_len: float,
    prior_wid: float,
    prior_h: float,
) -> float:
    """
    Estimate yaw (radians) from the 2D bounding box aspect ratio.

    Model: the projected width of a box-shaped object at yaw angle θ is
        visible_w = prior_len * |sin(θ)| + prior_wid * |cos(θ)|
    and the projected height ≈ prior_h (ignoring pitch/camera elevation).

    We search θ in [0, π/2] for the angle that best matches the observed
    pixel aspect ratio, then return it.  Front/rear ambiguity (θ vs π - θ)
    is unresolvable from a single frame; the tracker's EMA will smooth it.

    Returns a yaw in [0, π/2].
    """
    if bbox_h_px <= 0 or prior_h <= 0:
        return 0.0

    observed_ratio = bbox_w_px / bbox_h_px        # pixels
    expected_h_ratio = prior_h / prior_h           # = 1.0 (normalised)

    best_yaw = 0.0
    best_err = float("inf")

    for deg in range(0, 91):                       # 0° = front/rear, 90° = side
        theta = math.radians(deg)
        # Expected pixel width / pixel height at this yaw
        expected_w = prior_len * abs(math.sin(theta)) + prior_wid * abs(math.cos(theta))
        expected_ratio = expected_w / prior_h

        err = abs(expected_ratio - observed_ratio)
        if err < best_err:
            best_err = err
            best_yaw = theta

    return best_yaw


# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------

@dataclass
class Detection3D:
    position: np.ndarray   # (3,) in target frame
    size: tuple            # (length, width, height)
    yaw: float
    label: str
    depth_confidence: float  # 1.0 = good bbox height, <1.0 = small/uncertain box


# ---------------------------------------------------------------------------
# Constant-velocity Kalman tracker  (same as detection_2_5d)
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    x: np.ndarray       # [x, y, z, vx, vy, vz]
    P: np.ndarray
    age: int = 0
    hits: int = 1
    misses: int = 0
    label: str = ""
    yaw: float = 0.0
    size: tuple = (1.0, 1.0, 1.0)

    _q_pos: float = 0.5
    _q_vel: float = 2.0
    _r: float = 0.5       # slightly higher than LiDAR fusion — depth is noisier

    def predict(self, dt: float):
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.zeros((6, 6), dtype=np.float32)
        for i in range(3):
            Q[i, i] = self._q_pos * dt**2
            Q[i+3, i+3] = self._q_vel * dt**2
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

        alpha = 0.4
        dyaw = math.atan2(
            math.sin(det.yaw - self.yaw),
            math.cos(det.yaw - self.yaw),
        )
        self.yaw += alpha * dyaw
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.size = tuple(
            (1 - alpha) * s0 + alpha * s1
            for s0, s1 in zip(self.size, det.size)
        )

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]


class MultiObjectTracker:
    def __init__(self, max_misses: int = 5, gate_distance: float = 5.0):
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

        n_t, n_d = len(self.tracks), len(detections)

        if n_t > 0 and n_d > 0:
            cost = np.array([
                [np.linalg.norm(trk.position - det.position) for det in detections]
                for trk in self.tracks
            ], dtype=np.float32)

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < self.gate_distance:
                    self.tracks[r].update(detections[c])
                    matched_t.add(r)
                    matched_d.add(c)

            for i in range(n_t):
                if i not in matched_t:
                    self.tracks[i].misses += 1

            for j in range(n_d):
                if j not in matched_d:
                    self._create_track(detections[j])

        elif n_d > 0:
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
        P[0, 0] = P[1, 1] = P[2, 2] = 2.0   # higher initial uncertainty (no LiDAR)
        P[3, 3] = P[4, 4] = P[5, 5] = 10.0
        self.tracks.append(Track(
            track_id=self._next_id, x=x, P=P,
            label=det.label, yaw=det.yaw, size=det.size,
        ))
        self._next_id += 1


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class DetectionMono(Node):
    def __init__(self):
        super().__init__('detection_mono')

        # Topics & frames
        self.declare_parameter('cam_info_topic',       '/camera/camera/color/camera_info')
        self.declare_parameter('detections_topic',     '/camera/detections')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('target_frame',         'base_link')

        # Depth estimation
        self.declare_parameter('min_box_height_px',    10)    # discard tiny detections
        self.declare_parameter('max_depth_m',          60.0)  # clamp implausibly far estimates
        self.declare_parameter('fallback_depth_m',     15.0)  # used when box is too small

        # Yaw estimation
        self.declare_parameter('use_aspect_ratio_yaw', True)  # aspect-ratio heuristic for vehicles

        # Tracker
        self.declare_parameter('enable_tracking',      True)
        self.declare_parameter('tracker_max_misses',   5)
        self.declare_parameter('tracker_gate_m',       5.0)   # wider gate than LiDAR fusion
        self.declare_parameter('min_hits_to_publish',  2)

        # Visualization
        self.declare_parameter('marker_alpha',         0.6)
        self.declare_parameter('marker_lifetime_s',    0.3)

        # State
        self.have_cam_info = False
        self.fx = self.fy = self.cx = self.cy = None

        self.tracker = MultiObjectTracker(
            max_misses=int(self.get_parameter('tracker_max_misses').value),
            gate_distance=float(self.get_parameter('tracker_gate_m').value),
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        cam_info_topic = self.get_parameter('cam_info_topic').value
        det_topic = self.get_parameter('detections_topic').value

        self.create_subscription(CameraInfo, cam_info_topic, self._on_info, 10)
        self.create_subscription(Detection2DArray, det_topic, self._on_detections, 10)

        # Publisher
        self.pub_markers = self.create_publisher(MarkerArray, '/detection_boxes_3d_mono', 10)

        self.get_logger().info("Camera-only mono depth → 3D box node started")
        self.get_logger().info(f"  CameraInfo:  {cam_info_topic}")
        self.get_logger().info(f"  Detections:  {det_topic}")
        self.get_logger().info("  Output:      /detection_boxes_3d_mono")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_info(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.have_cam_info = True

    def _on_detections(self, msg: Detection2DArray):
        if not self.have_cam_info:
            return

        cam_frame = self.get_parameter('camera_optical_frame').value
        target_frame = self.get_parameter('target_frame').value
        min_box_h = int(self.get_parameter('min_box_height_px').value)
        max_depth = float(self.get_parameter('max_depth_m').value)
        fallback_depth = float(self.get_parameter('fallback_depth_m').value)
        use_aspect_yaw = bool(self.get_parameter('use_aspect_ratio_yaw').value)

        # TF: camera optical → target frame
        try:
            tf_base_cam = self.tf_buffer.lookup_transform(
                target_frame, cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

        stamp = msg.header.stamp
        detections_3d: list[Detection3D] = []

        for det in msg.detections:
            uc = float(det.bbox.center.position.x)
            vc = float(det.bbox.center.position.y)
            w  = float(det.bbox.size_x)
            h  = float(det.bbox.size_y)

            # Resolve label
            label = ""
            if det.results:
                raw = det.results[0].hypothesis.class_id
                try:
                    label = COCO_NAMES.get(int(raw), raw)
                except (ValueError, TypeError):
                    label = raw
            label_lower = label.lower()

            size_prior = SIZE_PRIORS.get(label_lower, DEFAULT_SIZE)
            prior_len, prior_wid, prior_h = size_prior

            # --- Depth via size-prior back-projection ---
            # Z = fy * real_height / pixel_height
            confidence = 1.0
            if h >= min_box_h:
                Z = self.fy * prior_h / h
                Z = float(np.clip(Z, 0.5, max_depth))
            else:
                # Box too small to trust (partially out of frame, or tiny object)
                Z = fallback_depth
                confidence = 0.3

            # Back-project 2D box centre into 3D camera frame
            X = (uc - self.cx) * Z / self.fx
            Y = (vc - self.cy) * Z / self.fy

            p_cam = PointStamped()
            p_cam.header.frame_id = cam_frame
            p_cam.header.stamp = stamp
            p_cam.point.x = float(X)
            p_cam.point.y = float(Y)
            p_cam.point.z = float(Z)

            p_base = do_transform_point(p_cam, tf_base_cam)
            pos = np.array([p_base.point.x, p_base.point.y, p_base.point.z],
                           dtype=np.float32)

            # --- Yaw estimation from bbox aspect ratio ---
            yaw = 0.0
            if use_aspect_yaw and label_lower in YAW_ESTIMATE_CLASSES and w > 0 and h > 0:
                yaw = estimate_yaw_from_aspect(w, h, prior_len, prior_wid, prior_h)

            detections_3d.append(Detection3D(
                position=pos, size=size_prior, yaw=yaw,
                label=label, depth_confidence=confidence,
            ))

        # Tracker
        if bool(self.get_parameter('enable_tracking').value):
            active_tracks = self.tracker.step(detections_3d, stamp)
        else:
            active_tracks = [
                Track(
                    track_id=i,
                    x=np.concatenate([d.position, np.zeros(3, dtype=np.float32)]),
                    P=np.eye(6, dtype=np.float32),
                    label=d.label, yaw=d.yaw, size=d.size, hits=999,
                )
                for i, d in enumerate(detections_3d)
            ]

        self._publish_markers(active_tracks, stamp, target_frame)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_markers(self, tracks: list[Track], stamp, target_frame: str):
        alpha = float(self.get_parameter('marker_alpha').value)
        lifetime_s = float(self.get_parameter('marker_lifetime_s').value)
        min_hits = int(self.get_parameter('min_hits_to_publish').value)

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

        marr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marr.markers.append(clear)

        for trk in tracks:
            if trk.hits < min_hits:
                continue

            length, width, height = trk.size
            qx, qy, qz, qw = yaw_to_quaternion(trk.yaw)
            r, g, b = colors.get(trk.label.lower(), default_color)

            # Box marker
            m = Marker()
            m.header.frame_id = target_frame
            m.header.stamp = stamp
            m.ns = "mono_boxes"
            m.id = trk.track_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(trk.position[0])
            m.pose.position.y = float(trk.position[1])
            m.pose.position.z = float(trk.position[2])
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw
            m.scale.x = length
            m.scale.y = width
            m.scale.z = height
            m.color.a = alpha
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.lifetime = rclpy.duration.Duration(seconds=lifetime_s).to_msg()
            marr.markers.append(m)

            # Label text marker
            txt = Marker()
            txt.header.frame_id = target_frame
            txt.header.stamp = stamp
            txt.ns = "mono_labels"
            txt.id = trk.track_id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(trk.position[0])
            txt.pose.position.y = float(trk.position[1])
            txt.pose.position.z = float(trk.position[2]) + height / 2.0 + 0.3
            txt.scale.z = 0.4
            txt.color.a = 1.0
            txt.color.r = txt.color.g = txt.color.b = 1.0
            txt.text = trk.label
            txt.lifetime = rclpy.duration.Duration(seconds=lifetime_s).to_msg()
            marr.markers.append(txt)

        self.pub_markers.publish(marr)


def main():
    rclpy.init()
    node = DetectionMono()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
