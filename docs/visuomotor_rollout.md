# Visuomotor Rollout
To test a visuomotor policy using the Trunk robot hardware, first set up the robot using and [motor control](./motor_control.md) and [video streaming](./video_streaming.md) instructions, then the following steps can be followed.

## Usage
Initialize a control solver node:
```bash 
cd asl_trunk_ws
source install/setup.bash
ros2 run controller ik_solver_node
```


To start the visuomotor rollout, run these commands in a new terminal:
```bash
cd asl_trunk_ws
source install/setup.bash
ros2 run executor visuomotor_node
```

The robot hardware will then react to vision input.

## Example Rollout
In the following video, a ResNet18 was trained to output desired trunk pose from an image of the robot enclosure. A set of ~90 training images were collected with AVP teleoperation to have the robot point toward the red octagon. Here's the rollout seen at 8x speed:

<iframe width="800" height="450" src="https://www.youtube.com/embed/5in-zQf-avg" frameborder="0" allowfullscreen></iframe>


## Visuomotor Policy Code
All code for training and testing visuomotor policies is available in this [Github repo](https://github.com/markeleone/trunk-visuomotor).