from setuptools import setup

package_name = 'teleop_twist_keyboard'

setup(
    name=package_name,
    version='2.4.0',
    packages=[],
    py_modules=[
        'teleop_twist_keyboard',
        'collisionavoidance',
        'detection',
        'newdetection',
        'aruco_controller',
        'repulsive_avoidance',
        'camera0_publisher',
        'camera1_publisher'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch.xml']),

    ],
    install_requires=['setuptools', 'torch', 'numpy', 'scipy', 'opencv-contrib-python', 'tf-transformations'],
    zip_safe=True,
    maintainer='Chris Lalancette',
    maintainer_email='clalancette@openrobotics.org',
    author='Graylin Trevor Jay, Austin Hendrix',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: BSD',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='A robot-agnostic teleoperation node to convert keyboard'
                'commands to Twist messages.',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_twist_keyboard = teleop_twist_keyboard:main',
            'collisionavoidance = collisionavoidance:main',
            'detection = detection:main',
            'newdetection = newdetection:main',
            'aruco_controller = aruco_controller:main',
            'repulsive_avoidance = repulsive_avoidance:main',
            'camera0_publisher = camera0_publisher:main',
            'camera1_publisher = camera1_publisher:main'
        ],
    },
)
