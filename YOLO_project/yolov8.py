import ultralytics
from ultralytics import YOLO
import cv2
import torch.nn as nn
import torch
from PIL import Image


class CustomLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(CustomLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
    def forward(self, x):
        return self.conv(x)

    # yaml_path = 'E:/Machine Learning/YOLO_Aug/YOLO_Aug.yaml'

# Load the model.
model_path = 'E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt'
# model = YOLO('E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt')
# image_path = "E:/Machine Learning/1507033044-1-16.mp4"
# prediction = model.predict(source=image_path, show=True)
# print(prediction)
yaml_path = "E:/Machine Learning/YOLO_Aug/YOLO_Aug.yaml"

#yolo_model = YOLO(model_path)

#custom_conv = CustomLayer()

#yolo_model.model.add_module()

#results_2 = yolo_model.train(
#   data=yaml_path,
#   epochs=50,
#   batch=8,
#   optimizer='Adam',
#   name='yolov8n_non_normalization'
#)

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

# 获取模型参数字典
model_params = yolo_model.state_dict()
print(model_params)
# 提取特定层权重
gamma_r = model_params['model.model.0.gamma_r']
gamma_g = model_params['model.model.0.gamma_g']
gamma_b = model_params['model.model.0.gamma_b']

color_kernel = model_params['model.model.1.conv.weight']
color_bias = model_params['model.model.1.conv.bias']

sharpen = model_params['model.model.2.conv.weight']
print(sharpen)