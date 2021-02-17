from model import autoencoder_model

import matplotlib.pyplot as plt

from keras import backend as K

import numpy as np
import Cluster as c
""" 프로젝트의 흐름을 제어하는 클래스 """
# 군집화
# 모델 훈련
# 모델 weight 저장 / 불러오기
# 군집화
# 성능 측정

class Agent(object):

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y
        self.model = autoencoder_model()

    def train_model(self, X_train, X_test, epochs):
        history = self.model.fit(X_train, X_train, epochs=epochs, validation_data=(X_test, X_test))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'], loc = 'upper left')
        plt.show()

    def test_model(self, compressed_layer=5):
        #테스트 진행할 전체 데이터 정규화
        X_ = self.X.astype('float32') / 255

        get_3rd_layer_output = K.function([self.model.layers[0].input],[self.model.layers[compressed_layer].output])

        compressed = get_3rd_layer_output([X_])[0]

        print(compressed.shape)
        return compressed

    def cluster_data(self, compressed, algorithm='kmeans'):
        compressed = compressed.reshape(compressed.shape[0],-1)

        result = []
        scaled_x = []
        if algorithm == 'kmeans':
            result, scaled_x = c.kmeans(compressed, 4, normalization='none')
        elif algorithm == 'dbscan':
            result, scaled_x = c.dbscan(compressed, normalization='none')
        
        return result, scaled_x
        
    def get_accuracy(self):
        
    def restore_img(self, num):
        # 모델 복구 (이미지화)
        restored_imgs = model.predict(X_test)

        for i in range(num):
            plt.imshow(X_test[i].reshape(28, 28))
            plt.gray()
            plt.show()
        
            plt.imshow(restored_imgs[i].reshape(28, 28))
            plt.gray()
            plt.show()

    def save_model(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, file_name):
        self.model.load_weights(file_name)
