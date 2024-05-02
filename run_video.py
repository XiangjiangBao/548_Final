import cv2
import time
from ultralytics import YOLO

model_path = ("E:/Machine Learning/yolov8n_custom_Aug_svd-20240428T225755Z-001/"
         "yolov8n_custom_Aug_svd/weights/best.pt")

original_path = ("F:/ML_Final/548_Final/weights/original.pt")
cover_path = ("F:/ML_Final/548_Final/weights/random_box.pt")
scale_path = ("F:/ML_Final/548_Final/weights/scale.pt")
rotate_illumination_path = ("F:/ML_Final/548_Final/weights/illu_rot.pt")

model = YOLO(model_path)
original_model = YOLO(original_path)
cover_model = YOLO(cover_path)
scale_model = YOLO(scale_path)
rotate_illumination_model = YOLO(rotate_illumination_path)

cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()
fps = 0
cv2.namedWindow("Video")



while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        prediction = model.predict(source=frame, show=False, verbose=False)
        color = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (255, 255, 0),
                 (255, 0, 255),
                 (0, 255, 255),
                 (122, 122, 0),
                 (0, 122, 122)]
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
            text_color = color[cls_index]
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            if center == []:
                center.append([(x1+x2)/2, (y1+y2)/2])
            else:
                center.append([(x1 + x2) / 2, (y1 + y2) / 2])
                if center[ii][0] - center[ii-1][0] <= 30 and center[ii][1] - center[ii-1][1] <= 30:
                    continue
            face_w = abs(float(x2-x1))/6
            face_h = abs(float(y2-y1))/6
            y1_n = max(y1-face_h, 0)
            y2_n = max(y2+face_h, 0)
            x1_n = max(x1 - face_w, 0)
            x2_n = max(x2 + face_w, 0)
            face = frame[int(y1_n):int(y2_n), int(x1_n):int(x2_n)]
            if text == 'Expression':
                r_prediction = original_model.predict(source=face, show=False, verbose=False)
            elif text == 'Occlution':
                r_prediction = cover_model.predict(source=face, show=False, verbose=False)
            elif text == 'Pose':
                r_prediction = rotate_illumination_model.predict(source=face, show=False, verbose=False)
            elif text == 'Illumination':
                r_prediction = rotate_illumination_model.predict(source=face, show=False, verbose=False)
            else:
                r_prediction = scale_model.predict(source=face, show=False, verbose=False)
            r_names = r_prediction[0].names
            expression_index = int(r_prediction[0].probs.top1)
            expression = r_names[expression_index]
            box_color = color[expression_index]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(img, text, (int(x1), int(y2) + 30), font, font_scale, text_color, 2)
            cv2.putText(img, expression, (int(x1), int(y1) - 10), font, font_scale, box_color, 2)

        if frame_count % 30 == 0 and frame_count >= 100:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
        cv2.putText(img, str(round(fps,2))+' fps', (10, 20), font, 0.5, color[0], 1)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("Video", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()