from setuptools import find_packages, setup

package_name = 'executor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/executor', ['executor/data_collection_node.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asl',
    maintainer_email='asl@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collection_node = executor.data_collection_node:main',
            'experiment_node = executor.experiment_node:main',
            'teleop_ik_node = executor.teleop_ik_node:main',
            'visuomotor_node = executor.visuomotor_node:main',
            'slider_node = executor.slider_node:main',
            'manual_decay_node = executor.manual_decay_node:main',
            'adiabatic_manual_decay_node = executor.adiabatic_manual_decay_node:main',
            'mpc_initializer_node = executor.mpc_initializer_node:main',
            'mpc_node = executor.mpc_node:main',
            'store_observations_node = executor.store_observations_node:main',
            'test_mpc_node = executor.test_mpc_node:main',
            'ffpid_node = executor.ffpid_node:main',
        ],
    },
)
