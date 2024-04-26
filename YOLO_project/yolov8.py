from ultralytics import YOLO
import cv2

yaml_path = 'E:/Machine Learning/YOLO_archive/YOLO.yaml'

# Load the model.
model = YOLO('yolov8n.pt')
results = model.train(
   data=yaml_path,
   epochs=50,
   batch=8,
   name='yolov8n_v8_50e'
)
