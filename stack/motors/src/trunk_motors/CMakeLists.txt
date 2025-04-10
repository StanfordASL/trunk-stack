cmake_minimum_required(VERSION 3.5)
project(trunk_motors)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################


###################################
## ament specific configuration  ##
###################################
ament_package()

#############
## Install ##
#############

# Install Python scripts
install(PROGRAMS
  trunk_motors/scripts/motor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY trunk_motors/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# find_package(ament_cmake_gtest REQUIRED)
# ament_add_gtest(${PROJECT_NAME}-test test/test_leap_hand.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()
