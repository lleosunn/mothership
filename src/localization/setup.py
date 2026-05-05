from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py') + glob('launch/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leosun',
    maintainer_email='leosun@todo.todo',
    description='Localization package for multi agent testbed',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher = localization.camera_publisher:main',
            'multi_camera_detection = localization.dual_camera_detection:main',
            'pose_logger = localization.pose_logger:main',
            'pose_visualizer = localization.pose_visualizer:main',
            'velocity_fusion = localization.velocity_fusion:main',
            'ekf = localization.ekf:main',
        ],
    },
)
