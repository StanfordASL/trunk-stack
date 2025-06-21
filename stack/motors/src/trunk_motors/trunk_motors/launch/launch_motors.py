import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch argument with default 'true'
        DeclareLaunchArgument(
            'secure_mode',
            default_value='true',
            description='Enable or disable secure mode for motor_node'
        ),
        Node(
            package='trunk_motors',
            executable='motor_node.py',
            name='motor_node',
            emulate_tty=True,
            output='screen',
            parameters=[
                {'kP': 800.0},
                {'kI': 0.0},
                {'kD': 0.0},
                {'curr_lim': 3000.0},
                {'secure_mode': LaunchConfiguration('secure_mode')}
            ]
        )
    ])
