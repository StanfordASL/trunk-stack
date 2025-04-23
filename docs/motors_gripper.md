# Motors and Gripper
The motor controllers are connected to the Raspberry Pi 4 (4GB RAM), which has Ubuntu 20.04 installed. The motor controllers are controlled using the [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) framework.
Optionally one can enable the servo-driven gripper as well.

## Usage
In the first terminal to start the motors, run:
```bash
cd motors/
source install/setup.bash
ros2 launch trunk_motors launch_motors.py
```

To start the gripper, in a another terminal, run:
```bash
cd gripper/
source install/setup.bash
ros2 run servo_control servo_control_node
```


