import csv
import grpc  # type: ignore
from threading import Thread
import numpy as np
import streamer.disktracking_pb2 as disktracking_pb2
import streamer.disktracking_pb2_grpc as disktracking_pb2_grpc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class AVPSubscriber:

    def __init__(self, ip, isRecording=False, gripper_open=False): 
        # Initialize with the Vision Pro IP
        self.ip = ip
        self.latest = None 
        self.last_error_message = None  
        self.isGripperOpen = gripper_open
        self.isRecording = isRecording 
        self.previousRecordingState = False
        self.start_streaming()

    def start_streaming(self): 
        stream_thread = Thread(target=self.stream)
        stream_thread.start() 
        while self.latest is None: 
            pass
        print('Ready to start streaming.') 

    def stream(self): 
        # Adjust the request type to your specific setup
        request = disktracking_pb2.DiskPosition(id='disk1', 
                                                position=[0.0, 0.0, 0.0])
        while True:
            try:
                with grpc.insecure_channel(f"{self.ip}:12345") as channel:
                    stub = disktracking_pb2_grpc.DiskTrackingServiceStub(channel)
                    responses = stub.StreamDiskPositions(request)
                    for response in responses:
                        # Transform and store the disk positions
                        positions = {
                            "disk_positions": {k: np.array(v.position) for k, v in response.disk_positions.items()},
                        }
                        self.previousRecordingState = self.isRecording
                        
                        self.isRecording = response.isRecording
                        self.isGripperOpen = response.isGripperOpen
                        self.latest = positions

                        # Force an operating frequency of 10 Hz
                        time.sleep(0.1)

            except Exception as e:
                if e.details() != self.last_error_message: # only print error message if its new
                    print(f"Connection failed, retrying: {e.details()}")
                    self.last_error_message = e.details()  
                pass 

    def get_latest(self): 
        return self.latest
        
    def get_recording(self): 
        return self.recording
    

class RealTimePlot3D:
    def __init__(self, streamer):
        self.streamer = streamer
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatters = {
            'disk1': self.ax.scatter([], [], [], label='Disk 1'),
            'disk2': self.ax.scatter([], [], [], label='Disk 2'),
            'disk3': self.ax.scatter([], [], [], label='Disk 3')
        }

        # Set axes limits and labels to match the desired orientation
        self.ax.set_xlim(-.5, .5)
        self.ax.set_ylim(-.5, .5)
        self.ax.set_zlim(-.5, 0)
        self.ax.set_xlabel('X', labelpad=20)
        self.ax.set_ylabel('Z', labelpad=20)
        self.ax.invert_yaxis()
        self.ax.set_zlabel('Y', labelpad=20)
        self.ax.set_title('Disk Positions in Real Time (3D)')
        self.ax.legend()

        # Adjust the view to match the desired axis orientation
        self.ax.view_init(elev=0, azim=90)

    def update(self, frame):
        latest_positions = self.streamer.get_latest()
        if latest_positions:
            for disk, scatter in self.scatters.items():
                if disk in latest_positions['disk_positions']:
                    pos = latest_positions['disk_positions'][disk]
                    scatter._offsets3d = (pos[0:1], pos[2:3], pos[1:2])  # Update x, y, z positions. x: 0:2, y: 1:2, z:2:3
                    # standard | ours
                    #     x    | x
                    #     y    | z
                    #     z    | y
        return self.scatters.values()

    def start(self):
        ani = FuncAnimation(self.fig, self.update, interval=100)
        plt.show()

if __name__ == "__main__":
    streamer = AVPSubscriber(ip='10.93.181.122', gripper_open=False, isRecording=False)
    plotter = RealTimePlot3D(streamer)
    plotter.start()
