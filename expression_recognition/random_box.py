# importing the general dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
from PIL import Image, ImageDraw
from scipy import misc
import random

def replace_with_random_box(image):
    # Get image size
    width, height = image.size
    # Define minimum and maximum box size ratios
    min_ratio, max_ratio = 0.2, 0.4
    # Calculate minimum and maximum box sizes
    min_box_size = min(width, height) * min_ratio
    max_box_size = min(width, height) * max_ratio
    # Randomly choose box size
    box_size = random.uniform(min_box_size, max_box_size)
    # Randomly choose box position
    x = random.randint(0, width - int(box_size))
    y = random.randint(0, height - int(box_size))
    # Randomly choose box color
    box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Replace the pixels within the box with the box color
    box = np.array([box_color], dtype=np.uint8)
    image_array = np.array(image)
    image_array[y:y+int(box_size), x:x+int(box_size)] = box
    # Convert the modified array back to an image
    modified_image = Image.fromarray(image_array)
    return modified_image

def replace_with_random_box_in_images(input_dir):
    # Function to replace parts of images with a random box and replace the original image
    def replace_with_random_box_image(image_path):
        img = Image.open(image_path)
        # Replace parts of the image with a random box
        modified_img = replace_with_random_box(img)
        # Save the modified image, overwriting the original image
        modified_img.save(image_path)

    # Loop through all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            replace_with_random_box_image(input_path)

    print("Images have been replaced with random boxes and replaced in", input_dir)


train_dir = 'E:/Machine Learning/archive_t/YOLO_format/train/images'
val_dir= 'E:/Machine Learning/archive_t/YOLO_format/valid/images'
test_dir = 'E:/Machine Learning/archive_t/YOLO_format/test/images'


replace_with_random_box_in_images(train_dir)
replace_with_random_box_in_images(val_dir)
replace_with_random_box_in_images(test_dir)