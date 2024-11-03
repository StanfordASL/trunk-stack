# Motors and Gripper
The motor controllers are connected to the Raspberry Pi 4 (4GB RAM), which has Ubuntu 20.04 installed. The motor controllers are controlled using the [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) framework.
Optionally one can enable the servo-driven gripper as well.

## Usage
In the first terminal, run:
```bash
cd motors/
sudo ./scripts/canableStart.sh  # start the CANable interface
source install/setup.bash
ros2 launch ros_phoenix trunk.launch.py
```
In the second terminal, run:
```bash
cd motors/
source install/setup.bash
ros2 run converter converter_node  # optionally add --ros-args -p debug:=true
```
Last, in a third terminal, run:
```bash
cd gripper/
source install/setup.bash
ros2 run servo_control servo_control_node
```

Now that the motors are set up, in the `main/` folder one can launch a slider interface to control each motor and the grippper:
```bash
cd main/
source install/setup.bash
ros2 run executor slider_node
```

## Motor control modes
The motor control modes are as follows:

| Mode                      | Value |
|---------------------------|-------|
| `PERCENT_OUTPUT`          | 0     |
| `POSITION`                | 1     |
| `VELOCITY`                | 2     |
| `CURRENT`                 | 3     |
| `FOLLOWER`                | 5     |
| `MOTION_PROFILE`          | 6     |
| `MOTION_MAGIC`            | 7     |
| `MOTION_PROFILE_ARC`      | 10    |

To simply run a command once to test the motor control, the following command can be used:

```bash
ros2 topic pub --once /all_motors_control interfaces/msg/AllMotorsControl "{motors_control: [{mode: 0, value: 0.25},{mode: 0, value: 0},{mode: 0, value: 0},{mode: 0, value: 0},{mode: 0, value: 0},{mode: 0, value: 0}]}"
```

## Motor control limits
The motor control limits are empirically established as follows:

$$
\operatorname{norm}\left(0.75\left(\vec{u}_3+\vec{u}_4\right)+1.0\left(\vec{u}_2+\vec{u}_5\right)+1.4\left(\vec{u}_1+\vec{u}_6\right)\right) \leq 0.6
$$

Going beyond this limit can result in the robot going outside of the workspace, which can be dangerous.
Therefore, we enforce this limit in our controllers.
