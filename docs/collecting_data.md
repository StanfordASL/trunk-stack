# Collecting Data
To collect data using the Trunk robot, after setting up the robot using the [motion capture](./mocap.md) and [motor control](./motor_control.md) instructions, the following steps can be followed.

## Usage
Essentially, all you need to run is contained in:
```bash
cd asl_trunk_ws
./scripts/run_data_collection.sh
```
Currently, this script will only collect steady-state data according to the control inputs as specified in [control_inputs.csv](https://github.com/hbuurmei/asl_trunk/blob/main/asl_trunk/asl_trunk_ws/data/trajectories/steady_state/control_inputs.csv) in the `asl_trunk_ws`.
The data will be saved in the `data/steady_state/` directory.
