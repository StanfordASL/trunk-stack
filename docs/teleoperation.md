# Teleoperation
To collect data for visuomotor policies using the Trunk robot with teleoperation, or to just teleoperate the robot for a demonstration, first set up the robot using the [motor control](./motor_control.md), [motion capture](./mocap.md), and [video streaming](./video_streaming.md) directions, then the following steps can be followed.

## Usage
Initialize a control solver node:
```bash
cd main/
source install/setup.bash
ros2 run controller ik_solver_node
```

Start an image storing node if you want to save the recorded data:
```bash
cd main/
source install/setup.bash
ros2 run streamer image_storing_node
```

Initialize the AVP streaming node:
```bash
cd main/
source install/setup.bash
ros2 run streamer avp_streamer_node
```

Finally, begin an executor node:
```bash
cd main/
source install/setup.bash
ros2 run executor run_teleop_ik_node 
```

You can then follow the prompts in the TrunkTeleop App on the AVP to calibrate the virtual trunk to the hardware trunk, then start streaming and recording data from trunk teleoperation. 

## Teleop Example
See the video below for an example of teleoperation in action.

<iframe width="800" height="450" src="https://www.youtube.com/embed/62IxsD0E3nY" frameborder="0" allowfullscreen></iframe>