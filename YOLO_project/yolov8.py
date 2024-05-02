import ultralytics
from ultralytics import YOLO
import cv2
from scipy import signal
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
def silu(x):
    return x * (1 / (1 + np.exp(-x)))


model_path = 'E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt'
# model = YOLO('E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt')
# image_path = "E:/Machine Learning/1507033044-1-16.mp4"
# prediction = model.predict(source=image_path, show=True)
# print(prediction)
yaml_path = "E:/Machine Learning/YOLO_Aug/YOLO_Aug.yaml"

file1 = ("E:/Machine Learning/"
        "yolov8n_Aug_1003-20240428T113208Z-001/"
        "yolov8n_Aug_1003/weights/best.pt")

file2=("E:/Machine Learning/yolov8n_cfg-20240428T073811Z-001/"
       "yolov8n_cfg/weights/best.pt")

file3 = ("E:/Machine Learning/yolov8n_custom-20240428T114813Z-001/"
         "yolov8n_custom/weights/best.pt")

file4 = ("E:/Machine Learning/yolov8n_custom_Aug_svd-20240428T225755Z-001/"
         "yolov8n_custom_Aug_svd/weights/best.pt")
yolo_model = YOLO(file4)
# add_custom_head(model.model)
data_path = "F:/ML_Final/548_Final/YOLO_Aug.yaml"
# yolo_model.val(data=data_path)
image_path = "F:/ML_Final/548_Final/0_Parade_marchingband_1_849.jpg"
model_params = yolo_model.state_dict()
gamma_r = model_params['model.model.0.gamma_r']
gamma_g = model_params['model.model.0.gamma_g']
gamma_b = model_params['model.model.0.gamma_b']

color_kernel = model_params['model.model.1.conv.weight']
color_bias = model_params['model.model.1.conv.bias']
sharpen = model_params['model.model.2.conv.weight']
print(sharpen[0][0])
image = cv2.imread(image_path)/255
image[:,:,0] = (image[:,:,0] * image[:,:,0]**gamma_r.tolist())
image[:,:,1] = (image[:,:,1] * image[:,:,1]**gamma_g.tolist())
image[:,:,2] = (image[:,:,2] * image[:,:,2]**gamma_b.tolist())
image[:,:,0] = (image[:,:,0] * float(color_kernel[0])+float(color_bias[0]))
image[:,:,1] = (image[:,:,1] * float(color_kernel[1])+float(color_bias[1]))
image[:,:,2] = (image[:,:,2] * float(color_kernel[2])+float(color_bias[2]))
image[:,:,0] = (signal.convolve2d(image[:,:,0],sharpen[0][0].tolist(), mode='same'))
image[:,:,1] = (signal.convolve2d(image[:,:,1],sharpen[1][0].tolist(), mode='same'))
image[:,:,2] = (signal.convolve2d(image[:,:,2],sharpen[2][0].tolist(), mode='same'))
image_p = (np.maximum(image,0)*512).astype(np.uint8)
image_n = (np.minimum(image,0)*512).astype(np.uint8)
image = (image+1)/2*255
image_int = image.astype(np.uint8)
cv2.imshow('window',image_n)
cv2.waitKey(0)
