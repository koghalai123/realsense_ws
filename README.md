# realsense_ws
One folder needs a python 3.11 venv. Its best to do this inside the folder 'realsense_ws_3_11': 

cd realsense_ws_3_11

python3.11 -m venv venv

source venv/bin/activate

pip install pyrealsense2 numpy pyzmq

The script can then be run:

python camera_publisher_3_11.py

# The other folder needs to be built:

cd realsense_ws_3_12

colcon build

source install/setup.bash

ros2 run ros2_realsense_implementation image_subscriber
