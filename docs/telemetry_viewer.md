# Telemetry Viewer

For visualizing telemetry data, such as the motor output currents, motor control temperatures, and webcam stream, we use the free [foxglove](https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ros2/) tool.

## Usage
You can run the Foxglove server/ROS2 node by running the following command in any ROS2 workspace:
```bash
ros2 launch foxglove_bridge foxglove_bridge.launch.xml
```
Note that Foxglove is installed for a particular ROS2 distribution, but you can install it for any distribution, see below.
Then, just open the Foxglove web interface via their website and connect.
You will be able to visualize almost any data type.
Finally, particular settings, such as topics to listen to, are stored and can be found [in the repo](https://github.com/hbuurmei/asl_trunk/tree/main/asl_trunk/asl_trunk_ws/foxglove).

## Installing Foxglove
The only thing to do to run Foxglove is to install the WebSocket server. This can be done by running the following command:
```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```
