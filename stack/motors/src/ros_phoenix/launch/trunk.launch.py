import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate trunk motors launch description with all motors."""
    container = ComposableNodeContainer(
        name="PhoenixContainer",
        namespace="",
        package="ros_phoenix",
        executable="phoenix_container",
        parameters=[{"interface": "can0"}],
        composable_node_descriptions=[
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon1",
                parameters=[{"id": 1, "P": 1.0, "D": 0.001, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon2",
                parameters=[{"id": 2, "P": 1, "D": 0.0, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon3",
                parameters=[{"id": 3, "P": 0.1, "D": 0.001, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon4",
                parameters=[{"id": 4, "P": 0.1, "D": 0.001, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon5",
                parameters=[{"id": 5, "P": 0.1, "D": 0.001, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
            ComposableNode(
                package="ros_phoenix",
                plugin="ros_phoenix::TalonSRX",
                name="talon6",
                parameters=[{"id": 6, "P": 0.1, "D": 0.001, "period_ms": 1000, "watchdog_ms": 2000}],
            ),
        ],
        output="screen",
    )

    return launch.LaunchDescription([container])
