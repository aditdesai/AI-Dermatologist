'''
This script loads the images from the melanoma_cancer_dataset directory, stores them into numpy arrays and saves them in the following files:

x_train.npy
y_train.npy
x_test.npy
y_test.npy
'''

import numpy as np
import os
import cv2


train_path = os.path.join(os.getcwd(), "melanoma_cancer_dataset/train/")
test_path = os.path.join(os.getcwd(), "melanoma_cancer_dataset/test/")


def load_folder(path):
    x_data = []
    y_data = []
    for folder in os.listdir(path):
        label = 1 if folder == "malignant" else 0
        folder_path = os.path.join(path, folder)

        for img in os.listdir(folder_path):
            image_path = os.path.join(folder_path, img)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            x_data.append(image)
            y_data.append(label)
    
    return np.array(x_data), np.array(y_data)


x_train, y_train = load_folder(train_path)
x_test, y_test = load_folder(test_path)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)