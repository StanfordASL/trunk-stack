# OptiTrack System

To obtain groundtruth position observations of the Trunk Robot, we use a motion capture system.
In particular, we use equipment purchased from [OptiTrack](https://www.optitrack.com/), which is one of the most well-known providers of such systems.

## System Components
- 4 [Flex 3](https://www.optitrack.com/cameras/flex-3/) motion capture cameras.
- 1 [OptiHub 2](https://optitrack.com/accessories/sync-networking/optihub/) for camera syncing and interfacing.
- 1 Windows machine for running OptiTrack's [Motive](https://optitrack.com/software/motive/) software.

## System Setup
The four cameras are installed in the bottom of the frame of the system, to ensure sufficient distance between the cameras and the Trunk body.
We have found that placing the cameras too close results in excessive 'ghost markers', i.e. extraneous reflections.
A minimum distance is unfortunately not provided by the manufacturer, but we have that approximately a minimum distance of 50-60 cm between the cameras and the body works well.

Once installed in the frame, one can directly connect the cameras to the OptiHub via USB, which is itself connected via USB to the Windows machine.

Note that the Windows machine needs to have a [Hardware Key](https://optitrack.com/accessories/license-keys/) to run the Motive software.

## Motive
The system uses Motive 2.3.7.
In this software, the marker location are captured and streamed over the network, to be picked up by ROS, as described in the [ROS2 Workspaces](./ros2_workspaces.md) page

First, make sure that the markers are installed in the correct locations as indicated in the [Trunk Body](./mechanical_design.md#trunk-body) section.
Then, verify that they are showing up in the Motive interface.
For each segment, simply select all the corresponding markers and create a rigid body for each segment.
Ensure that in the Streaming Pane, the OptiTrack Streaming Engine is on, and interfacing with the right network.
This should also have the rigid bodies enabled for streaming, as that is what we will be using as observations moving forward.

## Calibration
For accurate calibration of the system, a [Calibration Wand](https://optitrack.com/accessories/calibration-tools/) is recommended.
With the cameras being mounted sturdily to the frame, we have found that we rarely have to recalibrate.
