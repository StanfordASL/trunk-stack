import os
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from interfaces.msg import TrunkRigidBodies, AllMotorsControl, SingleMotorControl
from geometry_msgs.msg import Point
from interfaces.srv import ControlSolver
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from .data_aug import crop_image, plot_predictions_on_image
import torch 
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
# from .plot_tools import plot_predictions_on_image


# Future todo:
# - add mpc logic (uses mocap data)

class VisuomotorNode(Node):
    def __init__(self):
        super().__init__('visuomotor_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True
            ('controller_type', 'ik')                       # 'ik' or 'mpc'
        ])

        self.controller_type = self.get_parameter('controller_type').value
        self.debug = self.get_parameter('debug').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
        self.best_model_path = 'data/models/visuomotor/object_mocap_rb_regression/best_model.pth'
        self.latest_image = None
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.bridge = CvBridge()
        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
        self.recording_folder = os.path.join(self.data_dir, 'images')
        self.x3 = 0.01
        self.z3 = 0.01
        

        # Initialize 3D plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Set initial positions (replace with your desired values)
        self.plotter_positions = [
            [0.0, -0.1, 0.0],
            [0.0,  -0.2, 0.0],
            [0.0, -0.32, 0.0]
        ]
    

        # Create scatter plots
        self.scatter1 = self.ax.scatter([], [], [], label='Disk 1')
        self.scatter2 = self.ax.scatter([], [], [], label='Disk 2')
        self.scatter3 = self.ax.scatter([], [], [], label='Disk 3')

        # set plot settings
        # Set axes limits and labels to match the desired orientation
        self.ax.set_xlim(-.2, .2)
        self.ax.set_ylim(-.2, .2)
        self.ax.invert_yaxis()
        self.ax.set_zlim(-.32, 0)
        self.ax.set_xlabel('X', labelpad=20)
        self.ax.set_ylabel('Z', labelpad=20)
        self.ax.invert_yaxis()
        self.ax.set_zlabel('Y', labelpad=20)
        self.ax.set_title('Disk Positions in Real Time (3D)')
        self.ax.legend()
        # Adjust the view to match the desired axis orientation
        self.ax.view_init(elev=45, azim=285)

        # Start the animation
        plt.ion()
        plt.show()

        # define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # image input size to model (224,224)
            # out of memory error with 1080x1080, batch size 32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # load the ResNet
        self.model = self.load_model(self.model, self.best_model_path) 

        self.subscriber = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_subscriber_callback, 10)

        # initialize controller
        # need to add mpc logic here later
        if self.controller_type == 'ik':
            # Create control solver service client
            self.ctrl_client = self.create_client(
                ControlSolver,
                'ik_solver'
            )
            # Wait for service to become available
            while not self.ctrl_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, waiting...')
        else:
            raise ValueError('Invalid controller type: ' + self.controller_type + '. Valid options are: "ik" or "mpc". mpc not yet implemented')
        

        # Execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )

        # occurs at 10Hz
        self._timer = self.create_timer(1.0 / 10.0, self.vm_callback)
        self.get_logger().info('Visuomotor node has been started.')

    def update_plot(self):
        # Unpack positions into separate lists
        x = [pos[0] for pos in self.plotter_positions]
        y = [pos[1] for pos in self.plotter_positions]
        z = [pos[2] for pos in self.plotter_positions]

        # Clear previous scatter plots
        self.ax.clear()

        # Create new scatter plots with updated data
        self.scatter1 = self.ax.scatter(x[0], y[0], z[0], label='Disk 1')
        self.scatter2 = self.ax.scatter(x[1], y[1], z[1], label='Disk 2')
        self.scatter3 = self.ax.scatter(x[2], y[2], z[2], label='Disk 3')

        # Set plot limits and labels
        self.ax.set_xlim(-0.3, 0.3)
        self.ax.set_ylim(-0.3, 0.3)
        self.ax.invert_yaxis()
        self.ax.set_zlim(-0.32, 0)
        self.ax.set_xlabel('X', labelpad=20)
        self.ax.set_ylabel('Z', labelpad=20)
        self.ax.set_zlabel('Y', labelpad=20)
        self.ax.set_title('Disk Positions in Real Time (3D)')
        self.ax.legend()
        self.ax.view_init(elev=45, azim=285)

        # Redraw the plot
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # Short pause for responsiveness


    def vm_callback(self):
        # do inference at 10Hz
        if self.latest_image != None: #wait for an image to exist on the topic
            vm_positions = self.do_vision_inference(self.latest_image)

            trunk_rigid_bodies_msg = self.convert_model_positions_to_trunk_rigid_bodies(vm_positions)
            if self.debug:
                self.get_logger().info(f'VM model desired positions {vm_positions}')
            
            # Controller
            # Create request
            request = ControlSolver.Request()

            # Populate request with desired positions from teleop
            zf = np.array([coord for pos in trunk_rigid_bodies_msg.positions for coord in [pos.x, pos.y, pos.z]])
            self.plotter_positions = np.reshape(zf, (3, 3)).tolist()
            self.update_plot()

            # Center data around zero
            settled_positions = np.array([0, -0.10665, 0, 0, -0.20432, 0, 0, -0.320682, 0])
            zf_centered = zf - settled_positions
            self.x3 = zf_centered[6]
            self.z3 = zf_centered[8]
            request.zf = zf_centered.tolist()

            # Call the control service
            self.async_response = self.ctrl_client.call_async(request)
            self.async_response.add_done_callback(self.ctrl_service_callback)
    
    def ctrl_service_callback(self, async_response):
        # self.get_logger().info('ctrl service callback called')
        try:
            response = async_response.result()
            # self.get_logger().info('got a response')
            self.publish_control_inputs(response.uopt) #UNCOMMENT THIS
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def publish_control_inputs(self, control_inputs):
        control_message = AllMotorsControl()
        control_message.motors_control = [
            SingleMotorControl(mode=0, value=value) for value in control_inputs
        ]
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info(f'Published new motor control setting: {control_inputs}.')

    def load_model(self, model, best_model_path):
        # Replace the last fully connected layer to output 9 values (x1, y1, z1, x2, y2, z2, x3, y3, z3)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 9) #output layers is 9-vector

        # load the model weights
        model.load_state_dict(torch.load(best_model_path, weights_only=False, map_location=torch.device('cpu'))) # change on gpu

        # Move model to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Running on device: {device}")

        # Ensure that the model is correctly loaded on the CPU
        model = model.to('cpu')
        model.eval() #set model to evaluation mode
        return model

    def do_vision_inference(self, image):
        with torch.no_grad():
            return self.model(image) # should be a 9-vector

    def image_subscriber_callback(self, msg):
        # CHANGE CROP, RESIZE, and NORMALIZE DEPENDING ON TRAINING DATA
        latest_image_msg = msg

        # convert to cv2
        cv2_img = self.bridge.compressed_imgmsg_to_cv2(latest_image_msg, "rgb8") #replace with imgmsg_to_cv2 for non-compressed img

        img_name = 'sample.jpg'
        img_filename = os.path.join(self.recording_folder, img_name)
        cv2.imwrite(img_filename, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))

        processed_data_dir = 'data/trajectories/teleop/mocap_rb/processed/single_img_regression_mocap_rb.csv'
        #plot_predictions_on_image(self.x3, self.z3, img_filename, processed_data_dir)

        # crop image
        cropped_img = crop_image(cv2_img, left_pct=0.08, right_pct=0.05, top_pct=0, bottom_pct=0)

        img_name = 'cropped_sample.jpg'
        img_filename = os.path.join(self.recording_folder, img_name)
        cv2.imwrite(img_filename, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

        # convert to pillow
        img_pil = Image.fromarray(cropped_img)
        
        img_name = 'pil_sample.jpg'
        img_filename = os.path.join(self.recording_folder, img_name)
        img_pil.save(img_filename)

        # do transformations
        image_tensor = self.transform(img_pil)

        # Add batch dimension
        self.latest_image = image_tensor.unsqueeze(0)

    def convert_model_positions_to_trunk_rigid_bodies(self, model_positions):
        """
        Converts model positions from a CNN output to a ROS TrunkRigidBodies message.

        Args:
            model_positions: A tensor of shape (1, 9) representing the x, y, z coordinates of three points.

        Returns:
            A TrunkRigidBodies message containing the converted positions.
        """

        # Extract the coordinates from the tensor
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = model_positions[0].tolist()
        #self.get_logger().info(f'model_positions[0] = {model_positions[0]}')

        # Create a ROS message
        msg = TrunkRigidBodies()
        # msg.header.stamp = rospy.Time.now()
        # msg.header.frame_id = "base_link"  # Replace with the appropriate frame ID
        # msg.frame_number = 0  # Replace with the actual frame number
        msg.rigid_body_names = ["rigid_body1", "rigid_body2", "rigid_body3"]  # Replace with actual names
        msg.positions = [Point(x=x1, y=y1, z=z1), Point(x=x2, y=y2, z=z2), Point(x=x3, y=y3, z=z3)]

        return msg

def main(args=None):
    rclpy.init(args=args)
    visuomotor_node = VisuomotorNode()
    rclpy.spin(visuomotor_node)

    visuomotor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
