from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leosun',
    maintainer_email='leotsun01@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stay_in_place = control.stay_in_place:main',
            'repulsive_avoidance = control.repulsive_avoidance:main',
            'teleop_twist_keyboard = control.teleop_twist_keyboard:main',
        ],
    },
)
