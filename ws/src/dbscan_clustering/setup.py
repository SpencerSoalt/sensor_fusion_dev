from setuptools import setup
from glob import glob
import os

package_name = 'dbscan_clustering'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='DBSCAN 3D clustering node for fused LiDAR-camera point clouds.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dbscan_node = dbscan_clustering.dbscan_clustering:main',
        ],
    },
)
