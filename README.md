## This is the final project code for BME548

### Author:
##### Xiangjiang Bao (xiangjiang.bao@duke.edu)
##### Junfei Wang (junfei.wang@duke.duke)

### Content:
* Convert_data.py is the file used to convert the dataset into yolo format
* Show_image.py is the file used to perform data augmentation
* run_gif.py is the file used to generate prediction result for .gif or short videos
* run_videos.py is the file used to generate prediction result for PC camera capture
* yolov8.yaml is the file used to define the structure of customized yolo network
* weights/ directory contains the weights after training for yolo network

### How to run
* Please make sure you are using the environment provided in the following link:\
https://drive.google.com/file/d/1so5AAy0hiYaKFYiLDCyRDJKPkZXDRpLL/view?usp=sharing 
* In run_video.py and run_gif.py, please modify the `model_path` to the path to the weights
you want to use. The weights are stored in weights/ directory.
* After running run_video.py, the program will open the default camera of the device and perform detection.
* For other device option, please modify the code in run_video


### How to train
* The dataset used for training can be obtained from:/
https://drive.google.com/file/d/1NRnWEFJ_MFLFF5yZo8DaKX1KFjY_1bd3/view?usp=sharing \
https://drive.google.com/file/d/1eaYsbVetQACV-ofHOrxrdgop6woMBvul/view?usp=sharing 
* The file used to train the model can be found in train_file. The training was done in
Colab, hence modification of `path` in .yaml file for dataset is needed.
* Note that all the training and prediction __should__ run in the environment provided.



