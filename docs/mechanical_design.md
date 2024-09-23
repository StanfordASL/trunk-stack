# Mechanical Design
The ASL Trunk robot is a low cost, highly customizable, open-source desktop soft robot hardware platform. The trunk is powered by 6 motors, which control 12 tendons that terminate at 3 disks along the length of the robot. Custom pretensioning mechanisms keep the antagonistc tendons in tension, and the actuation unit routes the tendons into the trunk. 

[//]: # (TODO: self-link on this page and add citation, add BOM)

## Full BOM
The full working bill of materials is available [here](https://docs.google.com/spreadsheets/d/1P72TMokWnYh4jPumQLBwXQ3-0UVnGI-0Cx1MCqVu7Cc/edit?usp=sharing) . 

## Full CAD
<iframe src="https://stanford2289.autodesk360.com/shares/public/SH30dd5QT870c25f12fcd9883a938c317f7f?mode=embed" width="1024" height="768" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"  frameborder="0"></iframe>

## Trunk
The flexible body of the trunk is a standard vacuum hose, which was cut to length for our application. 3 custom 3d printed disks, which each have 12 radially symmetric channels, divide the trunk into 3 equal-length segments. At each disk, 4 tendons terminate. Each disk also has a unique arrangment of motion capture markers, so OptiTrack Motive can easily distinguish them from each other for pose estimation. A custom parallel jaw gripper, adapted from [this design](https://www.youtube.com/watch?v=Qfd0ikdnAsg), is mounted on the end effector, driven by a small servo housed within the trunk body. The jaws of the gripper are easily swappable for different applications, including carrying large amounts of weight (up to 600g).
<iframe src="https://stanford2289.autodesk360.com/shares/public/SH30dd5QT870c25f12fcc3451d73cc86f268?mode=embed" width="1024" height="768" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"  frameborder="0"></iframe>

## Actuation Unit
The actuation unit routes 12 tendons from their respective pretensioning mechanisms to the trunk. The main structure is a custom 3d printed mount, which connects to the frame. 6 entry holes with 6 corresponding shafts hold 12 small pulleys which route the tendons with minimal friction and no overlap. The bottom of the actuation unit has a snap-fit attachment for the top of the trunk.
<iframe src="https://stanford2289.autodesk360.com/shares/public/SH30dd5QT870c25f12fc22629b72247d4398?mode=embed" width="1024" height="768" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"  frameborder="0"></iframe>

## Pretensioning Mechanism (PTM)
The pretensioning mechanism is heavily inspired by [Yeshmukhametov et al., 2019](https://doi.org/10.3390/robotics8030051). A pretensioning mechanism is necessary to drive two antagonistic cables with the same motor, such that when one is pulled by the motor, the other does not go slack. Our design consists of a "sled" that passively tensions a tendon using two compression springs in series on the lower linear rail.
<iframe src="https://stanford2289.autodesk360.com/shares/public/SH30dd5QT870c25f12fc3f4bc1ea84dd9bed?mode=embed" width="800" height="600" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"  frameborder="0"></iframe>

## Motor Assemblies
Each motor assembly is centered around a CIM 12V motor. We use the CIM12V mount along with a custom 3d printed mount to securely attach the motor and Talon SRX controller to the frame. A custom 3d printed pulley is connected to the motor shaft using a shaft key, and the tendons are secured to the top of the pulley. The Talon encoder is mounted to the frame using a custom 3d printed mount.
<iframe src="https://stanford2289.autodesk360.com/shares/public/SH30dd5QT870c25f12fca5cfc22cb92d7282?mode=embed" width="1024" height="768" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"  frameborder="0"></iframe>