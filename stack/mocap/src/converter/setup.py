from setuptools import find_packages, setup

package_name = 'converter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/converter', ['converter/converter_node.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asl',
    maintainer_email='hugo.buurmeijer@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'converter_node = converter.converter_node:main'
        ],
    },
)
