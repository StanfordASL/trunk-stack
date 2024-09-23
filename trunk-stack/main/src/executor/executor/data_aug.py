import cv2 
import numpy as np
import os
import random
#import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def resize_image(image, target_size=(1080, 1080)):
    return cv2.resize(image, target_size)

def adjust_hue(image, hue_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_factor) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def adjust_value(image, value_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * value_factor
    hsv_image[:, :, 2][hsv_image[:, :, 2] > 255] = 255
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def adjust_bbox(bbox, image_shape, rotation_angle):
    height, width = image_shape[:2]
    c, x_center, y_center, w, h = bbox

    # Calculate new center based on rotation
    new_x_center, new_y_center = x_center * width, y_center * height
    if rotation_angle == 90:
        new_x_center, new_y_center = height - new_y_center, new_x_center
        temp = w
        w = h
        h = temp
    elif rotation_angle == 180:
        new_x_center, new_y_center = width - new_x_center, height - new_y_center
    elif rotation_angle == 270:
        new_x_center, new_y_center = new_y_center, width - new_x_center
        temp = w
        w = h
        h = temp

    # Convert back to normalized coordinates
    new_x_center /= height
    new_y_center /= width

    return [c, new_x_center, new_y_center, w, h]

def crop_image(image, left_pct, right_pct, top_pct, bottom_pct):
    height, width = image.shape[:2]

    # Calculate pixel values to crop
    left = int(width * left_pct)
    right = int(width * (1 - right_pct))
    top = int(height * top_pct)
    bottom = int(height * (1 - bottom_pct))

    # Crop the image using calculated pixel values
    cropped_image = image[top:bottom, left:right]

    return cropped_image

# use for val and test set
def crop_and_resize(image_path, output_dir):
    image = cv2.imread(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    # crop image to relevant region
    image = crop_image(image, left_pct=0.08, right_pct=0.05, top_pct=0, bottom_pct=0)
 
    # Resize image ( we want default to be 1080x1080)
    image = resize_image(image)
    cv2.imwrite(os.path.join(output_dir, f"{name}{ext}"), image)

# use for training set only
def augment_image(image_path, output_dir):
    image = cv2.imread(image_path)
    # print(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    # print(base_name, image.shape)

    # crop image to relevant region
    image = crop_image(image, left_pct=0.08, right_pct=0.05, top_pct=0, bottom_pct=0)
    # print(image.shape) #uncomment to see image size before scaling

    # Resize image ( we want default to be 1080x1080)
    image = resize_image(image)
    cv2.imwrite(os.path.join(output_dir, f"{name}{ext}"), image)

    # Random hue adjustment
    hue_factor = random.randint(-15, 15)
    hue_adjusted_image = adjust_hue(image, hue_factor)

    # Random value adjustment
    value_factor = random.uniform(0.5, 1.5)
    value_adjusted_image = adjust_value(hue_adjusted_image, value_factor)

    # Save augmented image and adjusted bounding boxes
    cv2.imwrite(os.path.join(output_dir, f"{name}_augmented{ext}"), value_adjusted_image)

    # return augmented filename
    return f"{name}_augmented{ext}"

def convert_to_pillow_coords(df, img_width, img_height, x_min, x_max, z_min, z_max):
    """
    Convert robot coordinates to Pillow image coordinates.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x' and 'z' columns in robot coordinates.
    img_width, img_height (int): Dimensions of the Pillow image.

    Returns:
    pd.DataFrame: DataFrame with 'img_x' and 'img_y' columns for Pillow image coordinates.
    """

    # Calculate the scaling factors for x and z coordinates
    x_scale = img_width / (x_max - x_min)
    z_scale = img_height / (z_max - z_min)

    # Calculate the shifts to center the robot's origin within the image
    x_shift = (x_max + x_min) / 2
    z_shift = (z_max + z_min) / 2

    # Convert robot coordinates to Pillow image coordinates
    df['img_x'] = (df['x'] - x_shift) * x_scale + img_width / 2
    df['img_y'] = (df['z'] - z_shift) * z_scale + img_height / 2

    # Invert the y-axis and x-axis to match Pillow's coordinate system (where (0, 0) is top-left)
    df['img_y'] = img_height - df['img_y']
    df['img_x'] = img_width - df['img_x']

    return df[['img_x', 'img_y']]

def plot_predictions_on_image(x3, z3, image_path, dataset_file):
    """
    Plot ground truth and predicted tip positions on images.

    Parameters:
    results (list): List of dictionaries containing img_filename, true_x, true_z, pred_x, pred_z
    data_dir (str): Directory containing the images.
    """
    
    # Open the image
    with Image.open(image_path) as img:
        img_width, img_height = img.size

        # Convert ground truth and predicted coordinates to image coordinates
        positions_df = pd.read_csv(dataset_file)


        # Convert entire DataFrame coordinates to Pillow image coordinates once
        overall_df = positions_df[['x3', 'z3']].rename(columns={'x3': 'x', 'z3': 'z'})

        #calculate extent of dataset for scaling to pillow coords
        x_min, x_max = overall_df['x'].min() - 0.01, overall_df['x'].max() + 0.01
        z_min, z_max = overall_df['z'].min() - 0.01, overall_df['z'].max() + 0.01

        pred_img_coords = convert_to_pillow_coords(pd.DataFrame({'x': [x3], 'z': [z3]}), img_width, img_height,  x_min, x_max, z_min, z_max)

        # Create a figure with the same dimensions as the image
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

        # Plot the image
        ax.imshow(img)

        # Plot the predicted tip position
        ax.scatter([pred_img_coords['img_x'][0]], [pred_img_coords['img_y'][0]], color='red', s=200, label='Prediction')

        # Remove axes for a cleaner output
        ax.axis('off')

        # Add a legend
        ax.legend()

        # Save the image with the original dimensions
        output_filename = os.path.join("data/images/predicted_sample.jpg")
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)