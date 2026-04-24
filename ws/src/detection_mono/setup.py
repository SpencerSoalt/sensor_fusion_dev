from setuptools import setup
from glob import glob
import os

package_name = "detection_mono"

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
    description="Camera-only mono depth estimation via size-prior back-projection.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "detection_mono = detection_mono.detection_mono:main",
        ],
    },
)
