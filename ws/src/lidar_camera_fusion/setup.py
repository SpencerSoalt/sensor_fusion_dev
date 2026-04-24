from setuptools import setup
from glob import glob
import os

package_name = "lidar_camera_fusion"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@todo.todo",
    description="LiDAR-camera fusion: projects PointCloud2 into 2D bounding boxes to produce Detection3DArray.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lidar_camera_fusion = lidar_camera_fusion.lidar_camera_fusion:main",
        ],
    },
)
