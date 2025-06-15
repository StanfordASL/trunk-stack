#TRUNK SOURCES (Patrick)
source /opt/ros/humble/setup.bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
sudo cpupower frequency-set --governor performance


