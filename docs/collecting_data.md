# Collecting Data
To collect data using the Trunk robot, after setting up the robot using the [motion capture](./mocap.md) and [motor control](./motor_control.md) instructions, the following steps can be followed.

## Usage
Essentially, all you need to run is contained in:
```bash
cd main/
source install/setup.bash
ros2 run executor data_collection_node 
```

ROS arguments can be used to change the parameters. For example, to run control trajectory data collection with a control inputs file named "control_inputs_controlled_1.csv" and saving the observations to "observations_controlled_1.csv" you would run:
```
ros2 run executor data_collection_node --ros-args -p data_subtype:=’controlled’ -p results_name:=’observations_controlled_1’ -p input_num:=1
```

The data will be saved in the `main/data/trajectories/` directory.
