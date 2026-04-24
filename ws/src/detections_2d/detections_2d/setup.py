from setuptools import setup
from glob import glob
import os

package_name = "detections_2d"

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
    description="YOLOv12 (Ultralytics YOLO12) 2D detector for two frontal cameras; publishes vision_msgs/Detection2DArray.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "serialized = detections_2d.serialized:main",
            "old_batch = detections_2d.old_batch:main",
            "batch_2d = detections_2d.batch_2d:main",
            "single_2d = detections_2d.single_2d:main"

        ],
    },
)
