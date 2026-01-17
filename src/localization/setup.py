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
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
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
            'camera0_publisher = localization.camera0_publisher:main',
            'camera1_publisher = localization.camera1_publisher:main',
            'dual_camera_detection = localization.dual_camera_detection:main',
            'pose_visualizer = localization.pose_visualizer:main',
            'velocity_fusion = localization.velocity_fusion:main',
            'ekf = localization.ekf:main',
        ],
    },
)
