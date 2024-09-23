#!/bin/bash
source install/setup.bash

export TRUNK_DATA=/home/asl/Documents/asl_trunk_ws/data

ros2 run executor data_collection_node
