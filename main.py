import os, re, glob
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input

from agent import Agent

groups_folder_path = './resources/imgs28x28/'
# AXISX 2, 5, 6, 7 CLAMP 2, 3, 5, 6
AXISX_categ = ["AXISX2", "AXISX5", "AXISX6", "AXISX7"]
CLAMP_categ = ["CLAMP2", "CLAMP3", "CLAMP5", "CLAMP6"]
num_classes = 4

image_w = 28
image_h = 28
X = []
Y = []

def setData_XY(groups_folder_path, category, image_w, image_h):
    X = []
    Y = []
    #data_path = os.listdir(groups_folder_path)
    #print(data_path)
    for filepath in category:
        print(filepath)
        filepath = groups_folder_path + filepath
        files = os.listdir(filepath)
        for data in files:
            img_path = "{0}/{1}".format(filepath, data) # 이미지 경로 확인
            # 이미지 열고 알맞은 사이즈로 저장하기 
            image = Image.open(img_path)
            image = image.convert('L')#흑백으로 맹글기
            #image = image.resize((image_w, image_h))
            img_data = np.array(image)
            img_data = img_data.reshape((image_w, image_h, 1))

            X.append(np.array(img_data))

            seg = data.split('_')[3][0]
            seg = int(seg)

            Y.append(seg)
    return X, Y

X, Y = setData_XY(groups_folder_path, CLAMP_categ, image_w, image_h)

X = np.array(X)
Y = np.array(Y)
print("{}, {}".format(X.shape, Y.shape))

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

print("{}, {}".format(X_train.shape, Y_train.shape))
print("{}, {}".format(X_test.shape, Y_test.shape))

agent = Agent(X, Y)

# 데이터 정규화 하기
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

agent.train_model(X_train, X_test, 5)