import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile  # type: ignore
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from interfaces.srv import TriggerImageSaving
import cv2 #type: ignore
import os

class ImageStoringNode(Node):
    def __init__(self):
        super().__init__('image_storing_node')
        self.latest_image = None
        self.subscriber = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_storing_callback, 10)
        # if you want the non-compressed image, use Image and /image_raw as arguments above instead
        self.bridge = CvBridge()
        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
        self.recording_folder = os.path.join(self.data_dir, 'trajectories/teleop/mocap_rb/images')
        self.sample_id = 0 # need to match sample ID here and in streamer node
        self.service = self.create_service(TriggerImageSaving, 'trigger_image_saving', self.handle_trigger_image_saving)
        self.get_logger().info('Image saving service has been created.')

    def image_storing_callback(self, msg):
        self.latest_image = msg

    def handle_trigger_image_saving(self, request, response):
        number_of_images = request.number_of_images  # TODO: use this? last N img to save
        response.success = True

        img_name = f'sample_{self.sample_id}.jpg'
        filename = os.path.join(self.recording_folder, img_name)
        response.img_filename = img_name

        cv2_img = self.bridge.compressed_imgmsg_to_cv2(self.latest_image, "rgb8") #replace with imgmsg_to_cv2 for non-compressed img
        cv2.imwrite(filename, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))

        self.get_logger().info(f"Saved image as {filename}")
        self.sample_id += 1
        return response


def main(args=None):
    rclpy.init(args=args)
    image_storing_node = ImageStoringNode()
    rclpy.spin(image_storing_node)
    image_storing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
