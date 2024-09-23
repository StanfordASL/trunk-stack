import grpc
from threading import Thread
import numpy as np
import disktracking_pb2
import disktracking_pb2_grpc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import csv

class DiskPositionStreamer:

    def __init__(self, ip, isRecording=False, gripper_open=False): 
        # Initialize with the Vision Pro IP
        self.ip = ip
        self.latest = None 
        self.last_error_message = None  
        self.isGripperOpen = gripper_open
        self.isRecording = isRecording 
        self.previousRecordingState = False
        self.data = []
        self.recording_id = -1
        self.csv_filename = "recorded_data.csv"
        self.initialize_csv()
        self.start_streaming()

    def initialize_csv(self):
        # Initialize the CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'isGripperOpen']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


    def start_streaming(self): 
        stream_thread = Thread(target=self.stream)
        stream_thread.start() 
        while self.latest is None: 
            pass 
        print(' == DATA IS FLOWING IN! ==')
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
                        self.isRecording = response.isRecording
                        self.isGripperOpen = response.isGripperOpen

                        # increment trajectory ID
                        if self.isRecording and not self.previousRecordingState:
                            self.recording_id += 1

                        if self.isRecording:
                            self.save_to_csv(positions)


                        self.previousRecordingState = self.isRecording
                        self.latest = positions 



            except Exception as e:
                if e.details() != self.last_error_message: # only print error message if its new
                    print(f"Connection failed, retrying: {e.details()}")
                    self.last_error_message = e.details()  
                pass 

    def save_to_csv(self, positions):
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [
                self.recording_id, 
                positions['disk_positions'].get('disk1', [0, 0, 0])[0],
                positions['disk_positions'].get('disk1', [0, 0, 0])[1],
                positions['disk_positions'].get('disk1', [0, 0, 0])[2],

                positions['disk_positions'].get('disk2', [0, 0, 0])[0],
                positions['disk_positions'].get('disk2', [0, 0, 0])[1],
                positions['disk_positions'].get('disk2', [0, 0, 0])[2],
                
                positions['disk_positions'].get('disk3', [0, 0, 0])[0],
                positions['disk_positions'].get('disk3', [0, 0, 0])[1],
                positions['disk_positions'].get('disk3', [0, 0, 0])[2],
                int(self.isGripperOpen)  # Convert boolean to 1 (True) or 0 (False)
            ]
            writer.writerow(row)

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
    streamer = DiskPositionStreamer(ip='10.93.181.122', gripper_open=False, isRecording=False)
    plotter = RealTimePlot3D(streamer)
    plotter.start()

