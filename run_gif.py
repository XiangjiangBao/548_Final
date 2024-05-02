import cv2
from ultralytics import YOLO  # 导入你训练好的模型
import imageio
model_path = 'E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt'
model = YOLO('E:/Machine Learning/detect-20240427T043656Z-001/detect/yolov8n_non_normalization/weights/best.pt')
gif_path = "E:/Machine Learning/test_yasi.mp4"

original_path = ("F:/ML_Final/548_Final/weights/original.pt")
original_model = YOLO(original_path)


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
             (0, 255, 255),
             (122, 122, 0),
             (0, 122, 122)]

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
        text_color = color[cls_index]
        x1 = float(box[0])
        y1 = float(box[1])
        x2 = float(box[2])
        y2 = float(box[3])
        if center == []:
            center.append([(x1 + x2) / 2, (y1 + y2) / 2])
        else:
            center.append([(x1 + x2) / 2, (y1 + y2) / 2])
            if center[ii][0] - center[ii - 1][0] <= 30 and center[ii][1] - center[ii - 1][1] <= 30:
                continue
        face_w = abs(float(x2 - x1)) / 6
        face_h = abs(float(y2 - y1)) / 6
        y1_n = max(y1 - face_h, 0)
        y2_n = max(y2 + face_h, 0)
        x1_n = max(x1 - face_w, 0)
        x2_n = max(x2 + face_w, 0)
        face = frame[int(y1_n):int(y2_n), int(x1_n):int(x2_n)]
        r_prediction = original_model.predict(source=face, show=False, verbose=False)
        r_names = r_prediction[0].names
        expression_index = int(r_prediction[0].probs.top1)
        expression = r_names[expression_index]
        box_color = color[expression_index]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(img, text, (int(x1), int(y2) + 30), font, font_scale, text_color, 2)
        cv2.putText(img, expression, (int(x1), int(y1) - 10), font, font_scale, box_color, 2)
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
imageio.mimsave(output_gif_path, frames, format='GIF', duration=0.005)
print(f"Saved as {output_gif_path}")
