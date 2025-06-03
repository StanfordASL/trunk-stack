# Motors
The 6 Dynamixel XM540-W150-R motors are connected via a U2D2 to the Linux machine. 

## Usage
REQUIRED: To set USB port latency to 1 ms instead of default 16 ms, run:
```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

In the same terminal, to start the motors, run:
```bash
cd motors/
source install/setup.bash
ros2 launch trunk_motors launch_motors.py
```



