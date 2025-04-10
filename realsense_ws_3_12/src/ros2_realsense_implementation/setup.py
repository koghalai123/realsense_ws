from setuptools import setup

package_name = 'ros2_realsense_implementation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='ROS2 implementation for receiving RealSense images via ZeroMQ',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_subscriber = ros2_realsense_implementation.image_subscriber:main',
            'image_display = ros2_realsense_implementation.image_display:main',

        ],
    },
)