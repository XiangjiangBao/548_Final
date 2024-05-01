import cv2
import time
from ultralytics import YOLO
model_path = ("E:/Machine Learning/yolov8n_custom_Aug_svd-20240428T225755Z-001/"
         "yolov8n_custom_Aug_svd/weights/best.pt")

_path

model = YOLO(model_path)



cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()
fps = 0
cv2.namedWindow("Video")

while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        prediction = model.predict(source=frame, show=False)
        color = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (255, 255, 0),
                 (255, 0, 255),
                 (0, 255, 255), ]
        img = frame
        h, w, _ = img.shape
        new_width = int(w * 1.5)
        new_height = int(h * 1.5)
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
                center.append([(x1+x2)/2, (y1+y2)/2])
            else:
                center.append([(x1 + x2) / 2, (y1 + y2) / 2])
                if center[ii][0] - center[ii-1][0] <= 30 and center[ii][1] - center[ii-1][1] <= 30:
                    continue
            face = frame[int(y1):int(y2), int(x1):int(x2)].copy()

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(img, text, (int(x1), int(y1) - 10), font, font_scale, box_color, 2)

        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
        cv2.putText(img, str(round(fps,2))+' fps', (10, 20), font, 0.5, color[0], 1)
        print("fps:", fps)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("Video", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()