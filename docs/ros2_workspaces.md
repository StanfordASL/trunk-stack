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
in the `main/` directory and subsequently source the new install:
```bash
source install/setup.bash
```
Check that the build process runs without any errors.

## Motion Capture Workspace
**Note:** This workspace runs on the main machine.

Make sure you have set up the motion capture system as described in the [OptiTrack System Setup](./optitrack.md) page.

Enter the `mocap/` folder. Most code is already present, except for the `mocap4ros2_optitrack` plugin, which can be installed by following the instructions on the [respective website](https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack).
For completeness, we also list the steps here.

Enter the `src/` directory.
Download the plugin's repository:
```
git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack.git
```
Install dependencies:
```
rosdep install --from-paths src --ignore-src -r -y
vcs import < mocap4ros2_optitrack/dependency_repos.repos
```
Now, make sure that configuration file, located in `mocap4ros2_optitrack/mocap4ros2_optitrack_driver/config/`, has the correct information.
Specifically, the `server_address` should be equal to the address that Motive is streaming to (see the Local Interface entry in the Streaming pane in Motive), and `local_address` should be the address of the main machine, that will be running this workspace.

Once that information is entered correctly, compile the workspace:
```bash
cd .. # enter 'mocap' dir.
colcon build --symlink-install
```
Do make sure that ROS2 Humble is sourced again before building. Certain warnings can come up but may be ignored.

Then, check that the Optitrack configuration works fine and is connected by running it once:
```bash
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
```
This should say "Configured!" as a last message.
As the driver node is a lifecycle node, you should transition to activate:
```
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
```
which should return "Transitioning successful".

## Motor Control Workspace
**Note:** This workspace runs on the Raspberry Pi.

For the control of the motors, we use the [ros_phoenix](https://github.com/vanderbiltrobotics/ros_phoenix) package.
Due to compatibility constraints of this package, we use ROS2 Foxy for this workspace.

Clone the `trunk-stack` repository on the Pi and enter the `motors/` folder.

