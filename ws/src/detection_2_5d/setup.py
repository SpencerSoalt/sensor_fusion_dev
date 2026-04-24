from setuptools import setup
from glob import glob
import os

package_name = "detection_2_5d"

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
    maintainer="Spencer Soalt",
    maintainer_email="sosoalt@ucsd.edu",
    description="2.5D detection: lifts 2D boxes into pseudo-3D using per-box LiDAR depth statistics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "detection_2_5d = detection_2_5d.detection_2_5d:main",
        ],
    },
)
