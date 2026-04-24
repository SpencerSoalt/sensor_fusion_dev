#!/usr/bin/env python3
"""
dbscan_node.py

Subscribes:
  /fused/cloud_in_boxes   (sensor_msgs/PointCloud2)
      Labelled point cloud from the fusion node.
      Expected fields: x, y, z, label (float32 cast of class_id)

Publishes:
  /detections_3d          (vision_msgs/Detection3DArray)
      Axis-aligned 3D bounding boxes with class labels from 2D detections.
  /dbscan/cluster_cloud   (sensor_msgs/PointCloud2)
      Colored cluster cloud for debug visualisation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN

from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, \
                             ObjectHypothesisWithPose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Point
import sensor_msgs_py.point_cloud2 as pc2


class DbscanNode(Node):
    def __init__(self):
        super().__init__('dbscan_node')

        #  Parameters 
        self.declare_parameter('eps',                   0.5)
        self.declare_parameter('min_samples',           5)
        self.declare_parameter('min_cluster_size',      5)
        self.declare_parameter('max_cluster_size',      2000)

        # Depth filtering
        self.declare_parameter('depth_filter_method',  'jump')   # 'jump' | 'percentile'
        self.declare_parameter('jump_threshold',        0.5)      # metres  (method: jump)
        self.declare_parameter('depth_percentile',      30.0)     # %       (method: percentile)
        self.declare_parameter('depth_tolerance',       1.3)      # multiplier on near-percentile depth

        self.declare_parameter('publish_cluster_cloud', True)

        # Same-class Cluster merging
        self.declare_parameter('cluster_merge_distance', 0.8)  # metres, box surface-to-surface

        self._read_params()

        #  Subscriber 
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            PointCloud2,
            '/fused/cloud_in_boxes',
            self.cloud_callback,
            qos,
        )

        #  Publishers 
        self.pub_dets    = self.create_publisher(Detection3DArray, '/detections_3d',        qos)
        self.pub_cluster = self.create_publisher(PointCloud2,      '/dbscan/cluster_cloud',  qos)
        self.pub_markers = self.create_publisher(MarkerArray,      '/dbscan/bbox_markers',   qos)

        self.get_logger().info(
            f'DBSCAN node ready | eps={self.eps} min_samples={self.min_samples} '
            f'depth_filter={self.depth_filter_method}'
        )

    def _read_params(self):
        self.eps                  = self.get_parameter('eps').value
        self.min_samples          = self.get_parameter('min_samples').value
        self.min_cluster_size     = self.get_parameter('min_cluster_size').value
        self.max_cluster_size     = self.get_parameter('max_cluster_size').value
        self.depth_filter_method  = self.get_parameter('depth_filter_method').value
        self.jump_threshold       = self.get_parameter('jump_threshold').value
        self.depth_percentile     = self.get_parameter('depth_percentile').value
        self.depth_tolerance      = self.get_parameter('depth_tolerance').value
        self.pub_cluster_cloud       = self.get_parameter('publish_cluster_cloud').value
        self.cluster_merge_distance  = self.get_parameter('cluster_merge_distance').value

    def cloud_callback(self, msg: PointCloud2):
        # Read cloud
        raw = list(pc2.read_points(msg, field_names=('x', 'y', 'z', 'label'), skip_nans=True))
        if len(raw) < self.min_samples:
            return

        data       = np.array(raw)                        # structured array
        xyz        = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
        raw_labels = data['label'].astype(np.int32)     # class_id per point

        # Per-class depth filtering (filter each class frustum independently)
        keep_mask = np.zeros(len(xyz), dtype=bool)

        for class_id in np.unique(raw_labels):
            cls_idx  = np.where(raw_labels == class_id)[0]
            cls_pts  = xyz[cls_idx]
            filtered = self._depth_filter(cls_pts)

            if len(filtered) == 0:
                continue

            # Map filtered points back — match by value since order is preserved
            # depth_filter returns a subset of cls_pts rows
            filtered_set = set(map(tuple, filtered.tolist()))
            for i, idx in enumerate(cls_idx):
                if tuple(cls_pts[i].tolist()) in filtered_set:
                    keep_mask[idx] = True

        xyz_filtered    = xyz[keep_mask]
        labels_filtered = raw_labels[keep_mask]

        if len(xyz_filtered) < self.min_samples:
            return

        # DBSCAN clustering 
        cluster_ids = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm='ball_tree',
            n_jobs=-1,
        ).fit_predict(xyz_filtered)

        # Collect valid raw clusters
        unique_clusters = set(cluster_ids) - {-1}
        raw_clusters    = []   # list of dicts: pts, cls_lbls, mins, maxes, cids

        for cid in unique_clusters:
            mask     = cluster_ids == cid
            pts      = xyz_filtered[mask]
            cls_lbls = labels_filtered[mask]

            if len(pts) < self.min_cluster_size or len(pts) > self.max_cluster_size:
                continue

            mins  = pts.min(axis=0)
            maxes = pts.max(axis=0)
            if np.any(maxes - mins < 0.05):
                continue

            raw_clusters.append({
                'pts':      pts,
                'cls_lbls': cls_lbls,
                'mins':     mins,
                'maxes':    maxes,
                'cids':     [cid],
            })

        # Merge nearby same-class clusters
        if self.cluster_merge_distance > 0.0 and len(raw_clusters) > 1:
            raw_clusters = self._merge_nearby_clusters(
                raw_clusters, self.cluster_merge_distance
            )

        # Build detections from clusters
        det_array        = Detection3DArray()
        det_array.header = msg.header
        cluster_colors  = {}   # raw DBSCAN cid → color, for debug cloud

        for i, cl in enumerate(raw_clusters):
            pts      = cl['pts']
            cls_lbls = cl['cls_lbls']
            mins     = cl['mins']
            maxes    = cl['maxes']
            center   = (mins + maxes) / 2.0
            dims     = maxes - mins

            if np.any(dims < 0.05):
                continue

            class_id = Counter(cls_lbls.tolist()).most_common(1)[0][0]
            color   = self._CLASS_COLORS.get(class_id, self._DEFAULT_COLOR)
            for cid in cl['cids']:
                cluster_colors[cid] = color

            det        = Detection3D()
            det.header = msg.header

            bbox                   = BoundingBox3D()
            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.position.z = float(center[2])
            bbox.size.x            = float(dims[0])
            bbox.size.y            = float(dims[1])
            bbox.size.z            = float(dims[2])
            det.bbox               = bbox

            hyp                     = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(class_id)
            hyp.hypothesis.score    = float(
                np.sum(cls_lbls == class_id) / len(cls_lbls)
            )
            det.results.append(hyp)
            det_array.detections.append(det)

        self.pub_dets.publish(det_array)
        self._publish_bbox_markers(det_array)

        self.get_logger().debug(
            f'Input pts: {len(xyz)} | After depth filter: {len(xyz_filtered)} | '
            f'Clusters: {len(unique_clusters)} | Valid dets: {len(det_array.detections)}'
        )

        # Debug cluster cloud
        if self.pub_cluster_cloud:
            self._publish_cluster_cloud(
                xyz_filtered, cluster_ids, cluster_colors, msg.header
            )


    #  Depth Filters
    def _depth_filter(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        if self.depth_filter_method == 'jump':
            return self._filter_jump(pts)
        elif self.depth_filter_method == 'percentile':
            return self._filter_percentile(pts)
        else:
            self.get_logger().warn(
                f'Unknown depth_filter_method "{self.depth_filter_method}", skipping filter.'
            )
            return pts

    #  Method 1: depth jump 
    def _filter_jump(self, pts: np.ndarray) -> np.ndarray:
        """
        Sort by depth, find the first gap > jump_threshold, discard beyond it.
        Best general-purpose method for outdoor LiDAR.
        """
        if len(pts) < 2:
            return pts

        depths     = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(depths)
        sorted_d   = depths[sorted_idx]

        diffs    = np.diff(sorted_d)
        jump_pos = np.argmax(diffs > self.jump_threshold)

        if diffs[jump_pos] > self.jump_threshold:
            keep = sorted_idx[:jump_pos + 1]
        else:
            keep = sorted_idx   # no discontinuity — keep all

        return pts[keep]

    #  Method 2: percentile truncation
    def _filter_percentile(self, pts: np.ndarray) -> np.ndarray:
        """
        Keep points within depth_tolerance × near-percentile depth.
        Fast and simple — good for real-time on embedded hardware.
        """
        depths = np.linalg.norm(pts, axis=1)
        near_d = np.percentile(depths, self.depth_percentile)
        return pts[depths <= near_d * self.depth_tolerance]


    #  CLuster Merging
    @staticmethod
    def _merge_nearby_clusters(clusters, merge_dist):
        """
        Union-find merge of same-class clusters whose bounding boxes are
        within `merge_dist` metres (surface-to-surface) of each other.
        """
        n      = len(clusters)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                ci = Counter(clusters[i]['cls_lbls'].tolist()).most_common(1)[0][0]
                cj = Counter(clusters[j]['cls_lbls'].tolist()).most_common(1)[0][0]
                if ci != cj:
                    continue

                # AABB surface-to-surface distance
                gap = np.maximum(
                    0.0,
                    np.maximum(clusters[i]['mins'], clusters[j]['mins']) -
                    np.minimum(clusters[i]['maxes'], clusters[j]['maxes'])
                )
                if float(np.linalg.norm(gap)) < merge_dist:
                    parent[find(i)] = find(j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        merged = []
        for idxs in groups.values():
            if len(idxs) == 1:
                merged.append(clusters[idxs[0]])
                continue
            all_pts  = np.vstack([clusters[k]['pts']      for k in idxs])
            all_lbls = np.concatenate([clusters[k]['cls_lbls'] for k in idxs])
            all_cids = [cid for k in idxs for cid in clusters[k]['cids']]
            merged.append({
                'pts':      all_pts,
                'cls_lbls': all_lbls,
                'mins':     all_pts.min(axis=0),
                'maxes':    all_pts.max(axis=0),
                'cids':     all_cids,
            })

        return merged

    #  Debug Bound Box Markers
    def _publish_bbox_markers(self, det_array: Detection3DArray):
        marker_array = MarkerArray()

        for i, det in enumerate(det_array.detections):
            b        = det.bbox
            class_id = int(det.results[0].hypothesis.class_id) if det.results else -1
            r, g, b_ = self._CLASS_COLORS.get(class_id, self._DEFAULT_COLOR)
            color   = (r / 255.0, g / 255.0, b_ / 255.0)

            # Wireframe cube
            box_marker              = Marker()
            box_marker.header       = det.header
            box_marker.ns           = 'dbscan_boxes'
            box_marker.id           = i * 2
            box_marker.type         = Marker.CUBE
            box_marker.action       = Marker.ADD
            box_marker.pose         = b.center
            box_marker.scale        = Vector3(x=b.size.x, y=b.size.y, z=b.size.z)
            box_marker.color        = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.25)
            box_marker.lifetime.sec = 1
            marker_array.markers.append(box_marker)

            # Wireframe outline (LINE_LIST of the 12 box edges)
            wire_marker              = Marker()
            wire_marker.header       = det.header
            wire_marker.ns           = 'dbscan_boxes'
            wire_marker.id           = i * 2 + 1
            wire_marker.type         = Marker.LINE_LIST
            wire_marker.action       = Marker.ADD
            wire_marker.pose         = b.center
            wire_marker.scale.x      = 0.04   # line width
            wire_marker.color        = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            wire_marker.lifetime.sec = 1

            hx, hy, hz = b.size.x / 2.0, b.size.y / 2.0, b.size.z / 2.0
            corners = [
                ( hx,  hy,  hz), ( hx,  hy, -hz),
                ( hx, -hy,  hz), ( hx, -hy, -hz),
                (-hx,  hy,  hz), (-hx,  hy, -hz),
                (-hx, -hy,  hz), (-hx, -hy, -hz),
            ]
            edges = [
                (0,1),(2,3),(4,5),(6,7),   # vertical edges
                (0,2),(1,3),(4,6),(5,7),   # horizontal edges (one face)
                (0,4),(1,5),(2,6),(3,7),   # depth edges
            ]
            for a, b_idx in edges:
                ca, cb = corners[a], corners[b_idx]
                wire_marker.points.append(Point(x=ca[0], y=ca[1], z=ca[2]))
                wire_marker.points.append(Point(x=cb[0], y=cb[1], z=cb[2]))

            marker_array.markers.append(wire_marker)

            # Class label text above the box
            if det.results:
                class_name = self._CLASS_NAMES.get(int(det.results[0].hypothesis.class_id), det.results[0].hypothesis.class_id)
                label_text = class_name
            else:
                label_text = '?'

            text_marker                          = Marker()
            text_marker.header                   = det.header
            text_marker.ns                       = 'dbscan_labels'
            text_marker.id                       = i
            text_marker.type                     = Marker.TEXT_VIEW_FACING
            text_marker.action                   = Marker.ADD
            text_marker.pose.position.x          = b.center.position.x
            text_marker.pose.position.y          = b.center.position.y
            text_marker.pose.position.z          = b.center.position.z + b.size.z / 2.0 + 0.15
            text_marker.pose.orientation.w       = 1.0
            text_marker.scale.z                  = 0.3   # text height in metres
            text_marker.color                    = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text                     = label_text
            text_marker.lifetime.sec             = 1
            marker_array.markers.append(text_marker)

        #  clear stale markers from previous detections.
        if not det_array.detections:
            for ns in ('dbscan_boxes', 'dbscan_labels'):
                clear = Marker()
                clear.header = det_array.header
                clear.ns     = ns
                clear.action = Marker.DELETEALL
                marker_array.markers.append(clear)

        self.pub_markers.publish(marker_array)

    #  Debug Point Cloud
    def _publish_cluster_cloud(self, xyz, cluster_ids, colors, header):
        fields = [
            PointField(name='x',  offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',  offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',  offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb',offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        rows = []
        for pt, cid in zip(xyz, cluster_ids):
            if cid == -1:
                r, g, b = 80, 80, 80       # noise → grey
            else:
                r, g, b = colors.get(cid, (255, 255, 255))

            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            rgb_f   = np.frombuffer(
                np.int32(rgb_int).tobytes(), dtype=np.float32
            )[0]
            rows.append([pt[0], pt[1], pt[2], rgb_f])

        if rows:
            cloud_msg = pc2.create_cloud(header, fields, rows)
            self.pub_cluster.publish(cloud_msg)



    # Class Mapping and colors for debug visualisation
    _CLASS_NAMES = {
        0: 'person',
        2: 'car',
        5: 'bus',
        7: 'truck',
    }

    # Fixed color per class (r, g, b) 0-255
    _CLASS_COLORS = {
        0: (255,  80,  80),   # person  → red
        2: ( 80, 220,  80),   # car     → green
        5: ( 80, 140, 255),   # bus     → blue
        7: (255, 165,   0),   # truck   → orange
    }
    _DEFAULT_COLOR = (200, 200, 200)  # unknown class → grey



def main(args=None):
    rclpy.init(args=args)
    node = DbscanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()