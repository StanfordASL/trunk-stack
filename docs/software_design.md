# Software Design

## Description

## ROS Graph
Generated using [ros2_graph](https://github.com/kiwicampus/ros2_graph/).

```mermaid
flowchart LR

/turtlesim:::main
/teleop_turtle:::node
/turtle1cmd_vel([/turtle1cmd_vel<br>geometry_msgs/msg/Twist]):::topic
/turtle1color_sensor([/turtle1color_sensor<br>turtlesim/msg/Color]):::bugged
/turtle1pose([/turtle1pose<br>turtlesim/msg/Pose]):::bugged
/clear[//clear<br>std_srvs/srv/Empty\]:::bugged
/kill[//kill<br>turtlesim/srv/Kill\]:::bugged
/reset[//reset<br>std_srvs/srv/Empty\]:::bugged
/spawn[//spawn<br>turtlesim/srv/Spawn\]:::bugged
/turtle1set_pen[//turtle1set_pen<br>turtlesim/srv/SetPen\]:::bugged
/turtle1teleport_absolute[//turtle1teleport_absolute<br>turtlesim/srv/TeleportAbsolute\]:::bugged
/turtle1teleport_relative[//turtle1teleport_relative<br>turtlesim/srv/TeleportRelative\]:::bugged
/turtle1/rotate_absolute{{/turtle1/rotate_absolute<br>turtlesim/action/RotateAbsolute}}:::action
/clear o-.-o /turtlesim
/kill o-.-o /turtlesim
/reset o-.-o /turtlesim
/spawn o-.-o /turtlesim
/turtle1set_pen o-.-o /turtlesim
/turtle1teleport_absolute o-.-o /turtlesim
/turtle1teleport_relative o-.-o /turtlesim
/teleop_turtle <==> /turtle1/rotate_absolute
/turtle1/rotate_absolute o==o /turtlesim
/turtle1cmd_vel --> /turtlesim
/turtlesim --> /turtle1color_sensor
/turtlesim --> /turtle1pose
/teleop_turtle --> /turtle1cmd_vel
subgraph keys[<b>Keys<b/>]
subgraph nodes[<b><b/>]
topicb((No connected)):::bugged
main_node:::main_node
end
subgraph connection[<b><b/>]
node1:::node
node2:::node
node1 o-. to server .-o service[/Service<br>service/Type\]:::service
service <-. to client .-> node2
node1 -- publish --> topic([Topic<br>topic/Type]):::topic
topic -- subscribe --> node2
node1 o== to server ==o action{{/Action<br>action/Type/}}:::action
action <== to client ==> node2
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
The Apple Vision Pro teleoperation app was written in Swift 5 using XCode Beta 16 for VisionOS 2.0 Beta. Beta versions of XCode and VisionOS were used since some functionality necessary for our app was only available in beta versions. 

Our 3D assets were programmatically generated with standard hierarchical RealityKit Entities. The entities are placed into a MeshResource.Skeleton, upon which a custom IKComponent is added. A corresponding IKSolver smoothly solves the inverse kinematics of the 3 spherical pendulum joints when the position of the end effector is commanded with a gesture. The disk selection gestures are created with DragGestures. The streaming functionality for our app was heavily inspired by [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop), using GRPC to stream disk positions over WiFi. 

Source code for the app can be found in this [GitHub repository](https://github.com/StanfordASL/trunk-teleop). 
