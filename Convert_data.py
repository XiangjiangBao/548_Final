from ultralytics import YOLO
import cv2
import shutil
import os
# Load the model.
# model = YOLO('yolov8n.pt')

# img = cv2.imread('F:/ML_Final/548_Final/0_Parade_marchingband_1_849.jpg')
# yo_1 = (x_ + x_ + w_) / 2 / w
# yo_2 = (y_ + y_ + h_) / 2 / h
# yo_3 = w_ / w
# yo_4 = h_ / h


def convert_box_num(h, w, x_, y_, w_, h_):
	yo_1 = (x_ + x_ + w_) / 2 / w
	yo_2 = (y_ + y_ + h_) / 2 / h
	yo_3 = w_ / w
	yo_4 = h_ / h
	return [round(yo_1, 5), round(yo_2, 5),
			round(yo_3, 5), round(yo_4, 5)]


def convert_classfication(c1, c2, c3, c4, c5, c6):
	classification = [c1, c2, c3, c4, c5, c6]
	result = classification.index(max(classification))
	return result


def convert_data_format(filename):
	obj_num = []
	entry_block = []
	in_file = open(filename, 'r')
	file = in_file.readlines()
	in_file.close()
	for num_line, line in enumerate(file):
		line = line.strip()
		if '.jpg' in line:
			image_name = line
			obj_number = int(file[num_line + 1].strip())
			obj_num.append(obj_number)
			if obj_number == 0:
				entry_block.append(['None'])
			else:
				sub_block = [image_name]
				for i in range(0, obj_number):
					sub_block.append(file[num_line + i + 2].strip())
				entry_block.append(sub_block)
	return entry_block


def convert_entry_block(root_path, entry_block):
	output_block = []
	for file_num, block in enumerate(entry_block):
		if block[0] == 'None':
			print(str(file_num) + ' Invalid')
			continue
		img = cv2.imread(root_path + block[0])
		h, w, _ = img.shape
		sub_block = []
		for line in block:
			if '.jpg' in line:
				sub_block.append(line)
			else:
				values = line.split()
				x_ = float(values[0])
				y_ = float(values[1])
				w_ = float(values[2])
				h_ = float(values[3])
				c1 = float(values[4])
				c2 = float(values[5])
				c3 = float(values[6])
				c4 = float(values[7])
				c5 = float(values[8])
				c6 = float(values[9])
				yo_num = convert_box_num(h, w, x_, y_, w_, h_)
				class_num = convert_classfication(c1, c2, c3, c4, c5, c6)
				sub_block.append([class_num, yo_num[0],
					  			  yo_num[1], yo_num[2], yo_num[3]])
		output_block.append(sub_block)
	return output_block


def output_file(path, output_block):
	for file_num, block in enumerate(output_block):
		filename_ = block[0].split('.')[0]
		filename__ =filename_.split('/')[1]
		filename = path + filename__ + '.txt'
		with open(filename, 'w') as file:
			for sublist in block:
				if '.jpg' not in sublist:
					line = ' '.join(map(str, sublist)) + '\n'
					file.write(line)


def main():
	val_root_path = ('E:/Machine Learning/YOLO_DATA/'
				 'WIDER_val/WIDER_val/images/')
	val_filename = ('F:/ML_Final/548_Final/'
				'wider_face_split/wider_face_split'
				'/wider_face_val_bbx_gt.txt')
	entry_block = convert_data_format(val_filename)
	output_block = convert_entry_block(val_root_path, entry_block)
	val_out_path = 'E:/Machine Learning/YOLO_archive/val/labels/'
	output_file(val_out_path, output_block)

	train_root_path = ('E:/Machine Learning/YOLO_DATA/'
				 'WIDER_train/WIDER_train/images/')
	train_filename = ('F:/ML_Final/548_Final/'
				'wider_face_split/wider_face_split'
				'/wider_face_train_bbx_gt.txt')
	entry_block = convert_data_format(train_filename)
	output_block = convert_entry_block(train_root_path, entry_block)
	train_out_path = 'E:/Machine Learning/YOLO_archive/train/labels/'
	output_file(train_out_path, output_block)


source_folder = ("E:/Machine Learning/"
               "YOLO_DATA/WIDER_test/"
               "WIDER_test/images/")
target_folder = ("E:/Machine Learning/"
                 "YOLO_archive/test/images")



if __name__ == "__main__":
    # for root, dirs, files in os.walk(source_folder):
    #     for file in files:
    #         source_file = os.path.join(root, file)
    #         target_file = os.path.join(target_folder, file)
    #         shutil.copy2(source_file, target_file)
	main()