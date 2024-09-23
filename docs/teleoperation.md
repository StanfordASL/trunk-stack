# Teleoperation
To collect data for visuomotor policies using the Trunk robot with teleoperation, or to just teleoperate the robot for a demonstration, first set up the robot using the [motor control](./motor_control.md), [motion capture](./mocap.md), and [video streaming](./video_streaming.md) directions, then the following steps can be followed.

## Usage
Initialize a control solver node:
```bash
cd asl_trunk_ws
source install/setup.bash
ros2 run controller ik_solver_node
```

Start an image storing node:
```bash
cd asl_trunk_ws
source install/setup.bash
ros2 run streamer image_storing_node
```

Start an executor node:
```bash
cd asl_trunk_ws
source install/setup.bash
ros2 run executor run_experiment_node 
```

Then initialize the AVP streaming node:
```bash
cd asl_trunk_ws
source install/setup.bash
ros2 run streamer avp_streamer_node
```

You can then follow the prompts in the TrunkTeleop App on the AVP to calibrate the virtual trunk to the hardware trunk, then start streaming and recording data from trunk teleoperation. 

## Teleop Example
See the video below for an example of teleoperation in action.

<iframe width="800" height="450" src="https://www.youtube.com/embed/62IxsD0E3nY" frameborder="0" allowfullscreen></iframe>