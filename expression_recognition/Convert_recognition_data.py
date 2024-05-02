import os
import shutil

folder_path = "E:/Machine Learning/archive_t/YOLO_format/valid/labels"
image_path = "E:/Machine Learning/archive_t/YOLO_format/valid/images"
anger_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Anger"
contempt_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Contempt"
disgust_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Disgust"
fear_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Fear"
happy_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Happy"
neutral_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Neutral"
sad_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Sad"
surprise_folder = "E:/Machine Learning/archive_t/YOLO_format/valid/Surprise"

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否以".txt"结尾
    if file_name.endswith(".txt"):
        name, extension = os.path.splitext(file_name)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            first_char = file.read(1)
            if file_name.startswith('ffhq'):
                image = name + ".png"
            else:
                image = name + ".jpg"
            image_p = os.path.join(image_path,image)
            print(image_p)
        if first_char == '0':
            if not os.path.exists(anger_folder):
                os.makedirs(anger_folder)
            shutil.copy(image_p, anger_folder)
        if first_char == '1':
            if not os.path.exists(contempt_folder):
                os.makedirs(contempt_folder)
            shutil.copy(image_p, contempt_folder)
        if first_char == '2':
            if not os.path.exists(disgust_folder):
                os.makedirs(disgust_folder)
            shutil.copy(image_p, disgust_folder)
        if first_char == '3':
            if not os.path.exists(fear_folder):
                os.makedirs(fear_folder)
            shutil.copy(image_p, fear_folder)
        if first_char == '4':
            if not os.path.exists(happy_folder):
                os.makedirs(happy_folder)
            shutil.copy(image_p, happy_folder)
        if first_char == '5':
            if not os.path.exists(neutral_folder):
                os.makedirs(neutral_folder)
            shutil.copy(image_p, neutral_folder)
        if first_char == '6':
            if not os.path.exists(sad_folder):
                os.makedirs(sad_folder)
            shutil.copy(image_p, sad_folder)
        if first_char == '7':
            if not os.path.exists(surprise_folder):
                os.makedirs(surprise_folder)
            shutil.copy(image_p, surprise_folder)





        # 输出第一个字符
