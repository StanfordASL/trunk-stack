#!/bin/bash

source /opt/ros/humble/setup.bash
source .bashrc

# Terminal 1: mocap driver
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack-ssmr/stack/mocap/
colcon build
source install/setup.bash
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
"

# Terminal 2: lifecycle activation + converter
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack-ssmr/stack/mocap/
colcon build
source install/setup.bash
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
ros2 run converter converter_node
"

# Terminal 3: motor launch
gnome-terminal -- bash -c "
source /opt/ros/humble/setup.bash
cd ~/Documents/trunk-stack-ssmr/stack/motors/
colcon build
source install/setup.bash
ros2 launch trunk_motors launch_motors.py
"

# Terminal 5: mocap driver
gnome-terminal -- bash -c "
cd ~/Documents/pbenito/gnn-mpc/
source /opt/ros/humble/setup.bash
exec bash
"


# Current terminal

cd ~/Documents/trunk-stack-ssmr/stack/main/
colcon build
source install/setup.bash
ros2 run executor socket_mpc_node
