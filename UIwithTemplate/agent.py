import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import os, re, glob
import cv2
import numpy as np
from model import CNNAutoEncoder_28, CNNAutoEncoder_96
from PIL import Image
from keras import backend as K
class Autoencoder_Agent(object):

    def __init__(self, model_size, optimizer,learning_rate):
        self.model_size = model_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if self.model_size == 28:
            self.model = CNNAutoEncoder_28(optimizer = optimizer, learning_rate = learning_rate)
            self.compressed_layer = 5
        else: 
            self.model = CNNAutoEncoder_96(optimizer = optimizer, learning_rate = learning_rate)
            self.compressed_layer = 8

    def train(self, X_train, batch_size, epochs, validation_data):
        self.model.fit(X_train,X_train,batch_size = batch_size,epochs=epochs,validation_data=(validation_data,validation_data))


    def feature_extract(self,X_data):
        get_5th_layer_output = K.function([self.model.layers[0].input],[self.model.layers[self.compressed_layer].output])
        compressed = get_5th_layer_output([X_data])[0]
        compressed = compressed.reshape(compressed.shape[0], compressed.shape[1]*compressed.shape[2]*compressed.shape[3])
        return compressed


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)