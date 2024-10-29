# Software Design

## Description
The software stack of the ASL Trunk Robot is primarily developed in ROS2, and can is divided into the components:
*main*, *motion capture*, *motor control*, *camera* and *gripper*.

We assume that several computing resources are available: a Rapsberry Pi for executing the motor control, a main linux machine for running the experiments, and a Windows machine for running the motion capture Motive software.

## ROS2 Graph
We provide an overall architecture of the ROS2 nodes, topics and services used in the software stack.
This diagram is inspired by [ros2_graph](https://github.com/kiwicampus/ros2_graph/).

```mermaid
flowchart 

/run_experiment_node[ /run_experiment_node ]:::main

/converter_node_motors[ /converter_node motors ]:::node
/converter_node_mocap[ /converter_node mocap ]:::node
/ros_phoenix_node[ /ros_phoenix_node ]:::node
%% /mocap4ros2_optitrack_node[ /mocap4ros2_optitrack_node ]:::node
%% /v4l2_camera_node[ /v4l2_camera_node ]:::node
/servo_control_node[ /servo_control_node ]:::node
/mpc_solver_node[ /mpc_solver_node ]:::node

/rigid_bodies([ /rigid_bodies ]):::topic
/markers([ /markers ]):::topic
/trunk_rigid_bodies([ /trunk_rigid_bodies ]):::topic
/trunk_markers([ /trunk_markers ]):::topic
/image_raw([ /image_raw ]):::topic
/talon1_set([ /talon1/set ]):::topic
/talon2_set([ /talon2/set ]):::topic
/talon3_set([ /talon3/set ]):::topic
/talon4_set([ /talon4/set ]):::topic
/talon5_set([ /talon5/set ]):::topic
/talon6_set([ /talon6/set ]):::topic
%% /talon1_status([ /talon1/status ]):::topic
%% /talon2_status([ /talon2/status ]):::topic
%% /talon3_status([ /talon3/status ]):::topic
%% /talon4_status([ /talon4/status ]):::topic
%% /talon5_status([ /talon5/status ]):::topic
%% /talon6_status([ /talon6/status ]):::topic
/all_motors_status([ /all_motors_status ]):::topic
/all_motors_control([ /all_motors_control ]):::topic

/move_gripper[/move_gripper\]:::service
/mpc_solver[/mpc_solver\]:::service
/ik_solver[/ik_solver\]:::service

/run_experiment_node --> /all_motors_control --> /converter_node_motors
/converter_node_motors --> /talon1_set --> /ros_phoenix_node
/converter_node_motors --> /talon2_set --> /ros_phoenix_node
/converter_node_motors --> /talon3_set --> /ros_phoenix_node
/converter_node_motors --> /talon4_set --> /ros_phoenix_node
/converter_node_motors --> /talon5_set --> /ros_phoenix_node
/converter_node_motors --> /talon6_set --> /ros_phoenix_node
%% /ros_phoenix_node --> /talon1_status --> /converter_node_motors
%% /ros_phoenix_node --> /talon2_status --> /converter_node_motors
%% /ros_phoenix_node --> /talon3_status --> /converter_node_motors
%% /ros_phoenix_node --> /talon4_status --> /converter_node_motors
%% /ros_phoenix_node --> /talon5_status --> /converter_node_motors
%% /ros_phoenix_node --> /talon6_status --> /converter_node_motors
/converter_node_motors --> /all_motors_status --> /run_experiment_node

/rigid_bodies --> /converter_node_mocap
/markers --> /converter_node_mocap
/converter_node_mocap --> /trunk_rigid_bodies --> /run_experiment_node
/converter_node_mocap --> /trunk_markers --> /run_experiment_node

%% /v4l2_camera_node --> /image_raw --> /run_experiment_node
/image_raw --> /run_experiment_node

/run_experiment_node <==> /move_gripper o==o /servo_control_node
/run_experiment_node <==> /mpc_solver o==o /mpc_solver_node 
/run_experiment_node <==> /ik_solver o==o /ik_solver_node 

subgraph keys[<b>Keys<b/>]
subgraph nodes[<b><b/>]
topicb((Not connected)):::bugged
main_node[main]:::main
end
subgraph connection[<b><b/>]
node1[node1]:::node
node2[node2]:::node
node1 o-.-o|to server| service[/Service\]:::service
service <-.->|to client| node2
node1 -->|publish| topic([Topic]):::topic
topic -->|subscribe| node2
node1 o==o|to server| action{{Action}}:::action
action <==>|to client| node2
end
end

classDef node opacity:0.9,fill:#2A0,stroke:#391,stroke-width:4px,color:#fff
classDef action opacity:0.9,fill:#66A,stroke:#225,stroke-width:2px,color:#fff
classDef service opacity:0.9,fill:#3B8062,stroke:#3B6062,stroke-width:2px,color:#fff
classDef topic opacity:0.9,fill:#852,stroke:#CCC,stroke-width:2px,color:#fff
classDef main opacity:0.9,fill:#059,stroke:#09F,stroke-width:4px,color:#fff
classDef bugged opacity:0.9,fill:#933,stroke:#800,stroke-width:2px,color:#fff
style keys opacity:0.15,fill:#FFF
style nodes opacity:0.15,fill:#FFF
style connection opacity:0.15,fill:#FFF
```

## Teleoperation

### Overview
The trunk robot can be teleoperated by a user wearing an Apple Vision Pro. We designed an augmented reality app written in Swift which initializes a virtual 3D, 3-link spherical pendulum overlayed on the real-world view of the user. Once the virtual trunk is initialized, the user can calibrate the position and orientation of the virtual trunk to the hardware system. After calibration, the user can look at one of the disks on the trunk, which then lights up to denote its selection. The user can pinch their thumb and forefinger to select the disk, then the position of the virtual disk will mirror the position of their hand. The virtual disk positions can optionally be streamed over WiFi to a ROS2 listener, which publishes the 3D positions of the 3 disks on the trunk to the desired positions topic. A controller node subscribes to this topic and calculates the motor outputs necessary to attain that pose. The updated motor outputs are published to the motors, which causes the hardware trunk to mirror the virtual trunk. Streaming of desired trunk positions is done at 10Hz, and all of the other ROS2 functions run at 100Hz.

### Swift App Design
The Apple Vision Pro teleoperation app was written in Swift 5 using XCode 16 for VisionOS 2.1.

Our 3D assets were programmatically generated with standard hierarchical RealityKit Entities. The entities are placed into a *MeshResource.Skeleton*, upon which a custom IKComponent is added. A corresponding IKSolver smoothly solves the inverse kinematics of the 3 spherical pendulum joints when the position of the end effector is commanded with a gesture. The disk selection gestures are created with DragGestures. The streaming functionality for our app was heavily inspired by [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop), using GRPC to stream disk positions over WiFi. 

Source code for the app can be found in this [GitHub repository](https://github.com/StanfordASL/trunk-teleop). 

