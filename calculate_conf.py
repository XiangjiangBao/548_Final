import cv2
import numpy as np
from ultralytics import YOLO  # 导入你训练好的模型
from matplotlib import pyplot as plt
import imageio
image_path = "F:/ML_Final/548_Final/photo.png"

original_path = ("F:/ML_Final/548_Final/weights/scale.pt")
original_model = YOLO(original_path)
imag = cv2.imread(image_path)
image = cv2.resize(imag, (500, 500))
print(image.shape)
w = image.shape[0]
h = image.shape[1]
length = 125
stride = 5
cropped = image.copy()
cropped[100:100 + length, 100:100 + length] = 0
prediction = original_model.predict(source=cropped.copy(), show=False, verbose=False)

cv2.imshow("Image", cropped)
cv2.waitKey(0)
conf_m = np.zeros([int((w-length)/stride+1),int((h-length)/stride+1)])
for i in range(0, int((w-length)/stride+1)):
    for j in range(0, int((h-length)/stride+1)):
        if j == 0:
            print(i)
        cropped = image.copy()
        cropped[i*stride:(i*stride+length), j*stride:(j*stride+length)] = 0
        prediction = original_model.predict(source=cropped, show=False, verbose=False)
        conf_m[i,j] = prediction[0].probs.data[5]

padding = int(length/2)
conf = cv2.resize(conf_m, (h-length, w-length))
padded = cv2.copyMakeBorder(conf, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

plt.imshow(padded)
plt.colorbar()
plt.show()
