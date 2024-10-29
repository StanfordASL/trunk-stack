# ROS2 Workspaces

Before setting up the ROS2 workspaces needed to control the robot, please make sure you have cloned this repository:
```bash
gh repo clone StanfordASL/trunk-stack
```
The ROS2 code lives in the `stack/` directory.

## Main Workspace
**Note:** This workspace runs on the main machine.

Enter the `main/` folder.
We primarily use ROS2 Humble, which should be sourced before doing the first build:
```bash
source /opt/ros/humble/setup.bash
```
For actually building the workspace, we use Colcon, i.e. run:
```bash
colcon build
```
in the `main/` directory and check that the build process runs without any errors.

??? note "General note"

    It is highly recommended to build individual packages (within a workspace) upon changes, using the syntax:
    ```bash
    colcon build --packages-select interfaces
    ```

## Motion Capture Workspace
**Note:** This workspace runs on the main machine.

Make sure you have set up the motion capture system as described in the [OptiTrack System Setup](./optitrack.md) page.

Enter the `mocap/` folder. Most code is already present, except for the `mocap4ros2_optitrack` plugin, which can be installed by following the instructions on the [respective website](https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack).
For completeness, we also list the steps here.

Enter the `src/` directory.
Download the plugin's repository:
```bash
cd src/
git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack.git
```
Install dependencies:
```bash
rosdep install --from-paths src --ignore-src -r -y
vcs import < mocap4ros2_optitrack/dependency_repos.repos
```
Now, make sure that configuration file, located in `mocap4ros2_optitrack/mocap4ros2_optitrack_driver/config/`, has the correct information.
Specifically, the `server_address` should be equal to the address that Motive is streaming to (see the Local Interface entry in the Streaming pane in Motive), and `local_address` should be the address of the main machine, that will be running this workspace.

Once that information is entered correctly, compile the workspace:
```bash
cd ..
colcon build --symlink-install
```
Do make sure that ROS2 Humble is sourced again before building. Certain warnings can come up but may be ignored.

Then, check that the Optitrack configuration works fine and is connected by running it once:
```bash
source install/setup.bash
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
```
This should say "Configured!" as a last message.
As the driver node is a lifecycle node, you should transition to activate by running in a separate terminal:
```bash
source install/setup.bash
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
```
which should return "Transitioning successful".

## Motor Control Workspace
**Note:** This workspace runs on the Raspberry Pi.

For the control of the motors, we use the [ros_phoenix](https://github.com/vanderbiltrobotics/ros_phoenix) package.
Due to compatibility constraints of this package, we use ROS2 Foxy for this workspace.

Clone the `trunk-stack` repository on the Pi and enter the `motors/` folder.

Enter the source directory and clone the package:
```bash
cd src/
git clone https://github.com/vanderbiltrobotics/ros_phoenix
```

Note that the launch script `trunk.launch.py` is also being version controlled, therefore ensure that is placed in its respective location.

Again, build the workspace:
```bash
cd ..
colcon build --symlink-install
```

## Camera Workspace
**Note:** This workspace runs on the Raspberry Pi.

First, change directory into the `camera/` folder.

We closely follow [this tutorial](https://medium.com/swlh/raspberry-pi-ros-2-camera-eef8f8b94304) to install the camera driver, with the difference being that the Pi has Ubuntu 20.04 installed, not Raspberry Pi OS (previously Raspbian).
Specifically, once ROS2 is installed, the following commands can be used to install the camera packages:
```bash
cd src/
git clone --branch foxy https://gitlab.com/boldhearts/ros2_v4l2_camera.git
git clone --branch foxy https://github.com/ros-perception/vision_opencv.git
git clone --branch foxy https://github.com/ros-perception/image_common.git
git clone --branch foxy-devel https://github.com/ros-perception/image_transport_plugins.git
cd ..
rosdep install --from-paths src -r -y
colcon build
```
You may need allow the camera to be accessed by the user, which can be done by adding the user to the `video` group, or by adding the following udev rule:
```bash
sudo nano /etc/udev/rules.d/99-webcam.rules
KERNEL=="video[0-9]*", MODE="0666"  # add this to the file
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Gripper Workspace
**Note:** This workspace runs on the Raspberry Pi.

The gripper workspace is the simplest workspace of all.
Change directory into the `gripper/` folder.

Make sure that the `pigpio` is installed on the Pi.
If this library is not installed yet, one can find a straightforward description [here](https://gist.github.com/tstellanova/8b1fb350a148eace6541b5fbd2c021ca).
Also the associated Python package should be installed:
```bash
pip install pigpio
```

Then, directly build the workspace:
```bash
colcon build
```
