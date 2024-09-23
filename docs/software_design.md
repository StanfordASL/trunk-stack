# Software Design

## ROS Graph

## Teleoperation

### Overview
The trunk robot is teleoperated by a user wearing an Apple Vision Pro. We designed an augmented reality app written in Swift which initializes a virtual 3d, 3-link spherical pendulum overlayed on the real-world view of the user. Once the virtual trunk is initialized, the user can calibrate the position and orientation of the virtual trunk to the hardware system. After calibration, the user can look at one of the disks on the trunk, which then lights up to denote its selection. The user can pinch their thumb and forefinger to select the disk, then the position of the virtual disk will mirror the position of their hand. The virtual disk positions can optionally be streamed over WiFi to a ROS listener, which publishes the 3d positions of the 3 disks on the trunk to the desired positions topic. A controller node subscribes to this topic and calculates the motor outputs necessary to attain that pose. The updated motor outputs are published to the motors, which causes the hardware trunk to mirror the virtual trunk. Streaming of desired trunk positions is done at 10Hz, and all of the other ROS functions run at 100Hz.

### Swift App Design
The Apple Vision Pro teleoperation app was written in Swift 5 using XCode Beta 16 for VisionOS 2.0 Beta. Beta versions of XCode and VisionOS were used since some functionality necessary for our app was only available in beta versions. 

Our 3D assets were programmatically generated with standard hierarchical RealityKit Entities. The entities are placed into a MeshResource.Skeleton, upon which a custom IKComponent is added. A corresponding IKSolver smoothly solves the inverse kinematics of the 3 spherical pendulum joints when the position of the end effector is commanded with a gesture. The disk selection gestures are created with DragGestures. The streaming functionality for our app was heavily inspired by [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop), using GRPC to stream disk positions over WiFi. 

Source code for the app can be found in this [GitHub repository](https://github.com/StanfordASL/trunk-teleop). 

