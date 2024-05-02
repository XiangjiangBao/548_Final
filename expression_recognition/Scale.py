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
    scale_factor = random.uniform(0.6, 0.8)
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def replace_with_scale_images(input_dir):
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


replace_with_scale_images(train_dir)
replace_with_scale_images(val_dir)
replace_with_scale_images(test_dir)