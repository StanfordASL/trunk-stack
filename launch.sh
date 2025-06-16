#!/bin/bash

# Terminal 1: mocap driver
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack/stack/mocap/
source install/setup.bash
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
"

# Terminal 2: lifecycle activation + converter
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack/stack/mocap/
source install/setup.bash
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
ros2 run converter converter_node
"

# Terminal 3: motor launch
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack/stack/motors/
source install/setup.bash
ros2 launch trunk_motors launch_motors.py
"

