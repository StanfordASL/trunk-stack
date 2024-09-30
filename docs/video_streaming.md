# Video Streaming
The Trunk robot is equipped with a camera that can be used to stream video data to a remote computer. This can be useful for teleoperation, data collection, or other applications. The video stream is published as a ROS2 topic, which can be subscribed to by other nodes in the ROS2 network.

## Usage
To start the video stream, run the following command on the PI:
```bash
cd cam_ws
source install/setup.bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:=[1920,1080]
```
This will start the video stream and publish the video data on the `/image` topic, and compressed video data on the `/image/compressed` topic.
To alter for instance the frame rate or resolution, simply add the arguments as:
```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:=[1920,1080] -p framerate:=15
```

To subscribe to the video stream, run the following command on the remote computer inside any ROS2 workspace:
```bash
ros2 run rqt_image_view rqt_image_view
```
and select the appropriate topic to view the video stream, e.g. `/image/theora`.
This can also be viewed directly with all the other data, as described in the [telemetry viewer](telemetry_viewer.md) page.
