# Motion Capture System
To obtain observations, we run the motion capture system.
We assume you are readily in the `stack` subdirectory of the `trunk-stack` repository.

## Usage
First, make sure the robot is turned on.
The motion capture cameras should show numbers 1-4.
The Windows laptop has to be connected to the OptiHub via USB, and be running the [Motive 2](https://docs.optitrack.com/v/v2.3) software.
Then, on the main computer run the following command:
```bash
cd mocap/
source install/setup.bash
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
```
and in a new terminal run:
```bash
cd mocap/
source install/setup.bash
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
ros2 run converter converter_node
```
You can choose whether to use *markers* or *rigid bodies* by changing the `type` parameter, i.e.
```bash
ros2 run converter converter_node --ros-args -p type:='markers'  # or 'rigid_bodies' (default)
```
