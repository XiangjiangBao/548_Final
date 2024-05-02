# importing the general dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
from PIL import Image
from scipy import misc


def random_color_shift(image):
    # Define a set of random colors
    random_colors = ['none', 'blue', 'red', 'green', 'yellow', 'cyan', 'magenta']
    # Select a random color from the set
    selected_color = np.random.choice(random_colors)
    # Define intensity range for the color shift
    min_intensity, max_intensity = 0.4, 1.2
    intensity = np.random.uniform(min_intensity, max_intensity)
    # Apply the random color shift
    image = change_light_color(image, color=selected_color, intensity=intensity)
    return image

def change_light_color(image, color='none', intensity=1.0):
    # Convert image to float32
    image = image.astype(np.float32)
    # Define color channels for different light colors
    if color == 'none':
        pass
    elif color == 'blue':
        # Increase blue channel
        image[:, :, 2] += 255 * intensity
    elif color == 'red':
        # Increase red channel
        image[:, :, 0] += 255 * intensity
    elif color == 'green':
        # Increase green channel
        image[:, :, 1] += 255 * intensity
    elif color == 'yellow':
        # Increase red and green channels
        image[:, :, 0:2] += 255 * intensity
    elif color == 'cyan':
        # Increase green and blue channels
        image[:, :, 1:] += 255 * intensity
    elif color == 'magenta':
        # Increase red and blue channels
        image[:, :, 0:3:2] += 255 * intensity
    else:
        raise ValueError("Unsupported color. Choose from 'blue', 'red', 'green', 'yellow', 'cyan', 'magenta'.")
        # Clip pixel values to [0, 255]
    image = np.clip(image, 0, 255)
    # Convert back to uint8
    image = image.astype(np.uint8)
    return image

import random

def rotate_and_color_shift_images(input_dir):
    # Function to rotate images randomly, apply color shift, and replace the original image
    def rotate_and_color_shift_image(image_path):
        img = Image.open(image_path)
        original_size = img.size  # Get original image size
        # Apply random color shift first
        img_array = np.array(img)
        img_array = random_color_shift(img_array)
        img = Image.fromarray(img_array)
        # Rotate the image
        angle = random.randint(0, 360)
        rotated_img = img.rotate(angle, expand=False)
        # Save the rotated and color-shifted image, overwriting the original image
        rotated_img.save(image_path)

    # Loop through all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            rotate_and_color_shift_image(input_path)

    print("Images have been rotated and color-shifted, and replaced in", input_dir)

train_dir = 'E:/Machine Learning/archive_t/YOLO_format/train/images'
val_dir= 'E:/Machine Learning/archive_t/YOLO_format/valid/images'
test_dir = 'E:/Machine Learning/archive_t/YOLO_format/test/images'


rotate_and_color_shift_images(train_dir)
rotate_and_color_shift_images(val_dir)
rotate_and_color_shift_images(test_dir)