import numpy as np
import random
import cv2
import os


def rename_label(txt_folder_path):
    folder_path = txt_folder_path
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            new_filename = 'aug_' + filename
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)


def add_noise(image, std, mean=0, add=1):
    if add == 0:
        return image
    # std = [0, 1] for data augmentation
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def gaussian_blur(image, length, sigmax=0, add=1):
    if add == 0:
        return image
    # length = [1, 45], length should be odd
    kernal_size = (length, length)
    blurred_image = cv2.GaussianBlur(image, kernal_size, sigmax)
    return blurred_image


def motion_blur(image, angle, length, add=1):
    if add == 0:
        return image
    # angle = [0, 360]; length = [1, 40]
    kernel_size = length
    kernel = np.zeros((kernel_size, kernel_size))
    angle_rad = angle * np.pi / 180
    center = (kernel_size - 1) / 2
    slope_tan = np.tan(angle_rad)
    slope_cot = 1 / slope_tan
    if abs(slope_tan) <= 1:
        for i in range(kernel_size):
            kernel[int(center + slope_tan * (i - center)), i] = 1
    else:
        for i in range(kernel_size):
            kernel[i, int(center + slope_cot * (i - center))] = 1
    kernel /= kernel.sum()
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def aug_image(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            status = [round(random.random()) for _ in range(3)]
            std = random.random()
            motion_angle = random.uniform(0, 360)
            motion_length = random.randint(1, 40)
            gaussion_length = random.randrange(1, 36, 2)
            image = add_noise(image, std, status[0])
            image = motion_blur(image, motion_angle, motion_length, status[1])
            image = gaussian_blur(image,gaussion_length ,status[2])

            output_path = os.path.join(output_folder, 'aug_' + filename)
            cv2.imwrite(output_path, image)  # 保存之前需要将像素值还原到0-255范围
            print(f"{filename} 处理完毕.")
    print("所有图像处理完毕.")


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


input_folder = 'E:/Machine Learning/YOLO_archive/train/images'
output_folder = 'E:/Machine Learning/YOLO_Aug/Aug_train/images'
txt_folder = 'E:/Machine Learning/YOLO_Aug/Aug_train/labels'
# aug_image(input_folder, output_folder)
rename_label(txt_folder)

