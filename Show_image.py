import cv2


def show_image(filename):
    image_path = 'E:/Machine Learning/YOLO_archive/train/images/'
    label_path = 'E:/Machine Learning/YOLO_archive/train/labels/'
    img = cv2.imread(image_path + filename + '.jpg')
    h, w, _ = img.shape
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    with open(label_path + filename + '.txt', 'r') as f:
        temp = f.readlines()
        for num, line in enumerate(temp):
            lines = line.strip()
            line = lines.split()
            x_, y_, w_, h_ = eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4])
            x1.append(w * x_ - 0.5 * w * w_)
            x2.append(w * x_ + 0.5 * w * w_)
            y1.append(h * y_ - 0.5 * h * h_)
            y2.append(h * y_ + 0.5 * h * h_)
            cv2.rectangle(img, (int(x1[num]), int(y1[num])), (int(x2[num]), int(y2[num])), (255, 0, 0))
            cv2.imshow('windows', img)
    cv2.waitKey(0)


show_image('1_Handshaking_Handshaking_1_145')