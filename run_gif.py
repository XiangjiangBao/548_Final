import cv2
from ultralytics import YOLO  # 导入你训练好的模型
import imageio
model_path = 'E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt'
model = YOLO('E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt')
gif_path = "F:/ML_Final/548_Final/test_gif.gif"
prediction = model.predict(source=gif_path, show=False)
gif = cv2.VideoCapture(gif_path)
frames = []

total_frames = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))

# for frame_index in range(total_frames):
while True:
    # 读取一帧
    ret, frame = gif.read()
    if not ret:
        break  # 如果读取失败，退出循环

    color = [(255, 0, 0),
             (0, 255, 0),
             (0, 0, 255),
             (255, 255, 0),
             (255, 0, 255),
             (0, 255, 255), ]
    img = frame
    prediction = model.predict(source=frame, show=False)
    h, w, _ = img.shape
    label = prediction[0].names
    cls = prediction[0].boxes.cls
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    center = []
    for ii, box in enumerate(prediction[0].boxes.xyxy):
        cls_index = int(cls[ii])
        text = label[cls_index]
        box_color = color[cls_index]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        if center == []:
            center.append([(x1 + x2) / 2, (y1 + y2) / 2])
        else:
            center.append([(x1 + x2) / 2, (y1 + y2) / 2])
            if center[ii][0] - center[ii - 1][0] <= 30 and center[ii][1] - center[ii - 1][1] <= 30:
                continue
        face = frame[int(y1):int(y2), int(x1):int(x2)].copy()

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(img, text, (int(x1), int(y2) + 10), font, 0.4, box_color, 1)
    frames.append(frame)
    cv2.imshow('Frame', frame)
    delay = 100
    key = cv2.waitKey(delay)

    # 检查是否按下了'q'键，如果按下了则退出循环
    if key == ord('q'):
        break

gif.release()
cv2.destroyAllWindows()

output_gif_path = 'F:/ML_Final/548_Final/out_gif.gif'
imageio.mimsave(output_gif_path, frames, format='GIF', duration=0.1)
print(f"Saved as {output_gif_path}")
