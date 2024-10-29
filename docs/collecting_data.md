# Collecting Data
To collect data using the Trunk robot, after setting up the robot using the [motion capture](./mocap.md) and [motor control](./motor_control.md) instructions, the following steps can be followed.

## Usage
Essentially, all you need to run is contained in:
```bash
cd main/
source install/setup.bash
ros2 run executor data_collection_node
```
The data will be saved in the `main/data/trajectories/` directory.
