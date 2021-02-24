import os, re, glob
import cv2
import numpy as np
#from keras.preprocessing import image
from PIL import Image

groups_folder_path = './resources/imgs300x300/'
categories = ["ASIS2", "ASIS5", "ASIS6", "ASIS7", "CLAMP2", "CLAMP3", "CLAMP5", "CLAMP6"]
num_classes = len(categories)

X = []
Y = []

data_path = os.listdir(groups_folder_path)
print(data_path)
for filepath in data_path:
    print(filepath)
    filepath = groups_folder_path + filepath
    file_path = os.listdir(filepath)
    # print(file_path)

    for data in file_path:
        path_file = filepath + '/' + data
        print(path_file)
        # print(data)
        # img = img = image.load_img(path_file, target_size=(28, 28,1))
        image = Image.open(path_file)
        image = image.convert('L')  # 흑백으로 맹글기
        image = image.resize((28, 28))
        img_data = np.array(image)
        img_data = img_data.reshape((28, 28, 1))

        ##img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h.shape[0])
        # img_data = image.img_to_array(img)
        # img_data = np.expand_dims(img_data, axis=0)
        # img_data = preprocess_input(img_data)
        # X.append(np.array(img_data))
        img_data_np = np.array(img_data)

        if '_AXISX_2' in data:
            X.append(img_data_np)
            # Y.append(categories[0])
        elif '_AXISX_5' in data:
            X.append(img_data_np)
            # Y.append(categories[1])
        elif '_AXISX_6' in data:
            X.append(img_data_np)
            # Y.append(categories[2])
        elif '_AXISX_7' in data:
            X.append(img_data_np)
            # Y.append(categories[3])
        elif '_CLAMP_2' in data:
            X.append(img_data_np)
            # Y.append(categories[4])
        elif '_CLAMP_3' in data:
            X.append(img_data_np)
            # Y.append(categories[5])
        elif '_CLAMP_5' in data:
            X.append(img_data_np)
            # Y.append(categories[6])
        elif '_CLAMP_6' in data:
            X.append(img_data_np)
            # Y.append(categories[7])

        if '_AXISX_2' in data:
            # X.append(img_data_np)
            Y.append(categories[0])
        elif '_AXISX_5' in data:
            # X.append(img_data_np)
            Y.append(categories[1])
        elif '_AXISX_6' in data:
            # X.append(img_data_np)
            Y.append(categories[2])
        elif '_AXISX_7' in data:
            # X.append(img_data_np)
            Y.append(categories[3])
        elif '_CLAMP_2' in data:
            # X.append(img_data_np)
            Y.append(categories[4])
        elif '_CLAMP_3' in data:
            # X.append(img_data_np)
            Y.append(categories[5])
        elif '_CLAMP_5' in data:
            # X.append(img_data_np)
            Y.append(categories[6])
        elif '_CLAMP_6' in data:
            # X.append(img_data_np)
            Y.append(categories[7])
        print(filepath)

#print(X)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./img_data.npy", xy)

from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model

X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle = True)

#정규화
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D

input_shape = (28, 28, 1)
model = Sequential()

# 1st convolution layer
model.add(Conv2D(16, (3, 3)  # 16 is number of filters and (3, 3) is the size of the filter.
                 , padding='same', input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# 2nd convolution layer
model.add(Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# here compressed version

# 3rd convolution layer
model.add(Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

# 4rd convolution layer
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Activation('sigmoid'))
model.summary()

#학습
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, X_train, epochs=1000, validation_data=(X_test, X_test))

#테스트 진행할 전체 데이터 정규화
X = X.astype('float32') / 255

from keras import backend as K
compressed_layer = 5
get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[compressed_layer].output])
compressed = get_3rd_layer_output([X])[0]

print(compressed.shape)

#일렬로 늘리기
compressed = compressed.reshape(1520,7*7*2)
print(compressed.shape)

# Load Dataset
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

# Needed Library!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 삽입!
import Cluster as c

result, scaled_x = c.kmeans(compressed, 8, normalization='minmax')
#result.labels_ = categories
#result, scaled_x = c.kmeans(compressed, 8, normalization='minmax')

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
#0 : ASIS2, 1 : ASIS5, 2 : ASIS6, 3 : ASIS7, 4 : CLAMP2, 5 : CLAMP3, 6 : CLAMP5, 7 : CLAMP6
print(result.labels_)

#c.visualization_clusters(compressed, result.labels_, ['ASIS2', 'ASIS5', 'ASIS6', 'ASIS7', 'CLAMP2', 'CLAMP3', 'CLAMP5', 'CLAMP6'])

"""
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 7 2 7 7 7 7 2
 7 7 7 7 2 2 2 7 7 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 7 7 7 7 7 7 7 7 7 7 7 2 2 7 2 7 7 7 7 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5]
 """
