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

## Re-installing
For instance, [this tutorial](https://medium.com/swlh/raspberry-pi-ros-2-camera-eef8f8b94304) can be followed to re-install the camera driver.
However, note that we installed Ubuntu 20.04 on the Pi, not Raspberry Pi OS (previously Raspbian), such that the ROS2 installation is different (much simpler).
Specifically, once ROS2 is installed, the following commands can be used to install the camera packages:
```bash
mkdir -p Documents/cam_ws/src && cd Documents/cam_ws/src
git clone --branch foxy https://gitlab.com/boldhearts/ros2_v4l2_camera.git
git clone --branch foxy https://github.com/ros-perception/vision_opencv.git
git clone --branch foxy https://github.com/ros-perception/image_common.git
git clone --branch foxy-devel https://github.com/ros-perception/image_transport_plugins.git
cd ..
rosdep install --from-paths src -r -y
colcon build
source install/setup.bash
```
Then, the camera can be started as described above.
You may need allow the camera to be accessed by the user, which can be done by adding the user to the `video` group, or by adding the following udev rule:
```bash
sudo nano /etc/udev/rules.d/99-webcam.rules
KERNEL=="video[0-9]*", MODE="0666"  # add this to the file
sudo udevadm control --reload-rules
sudo udevadm trigger
```
