from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
#from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from utils import optimizer_set

def CNNAutoEncoder_28(optimizer,learning_rate,activation='sigmoid', loss='mse'):
    optimizer = optimizer_set(optimizer, learning_rate)
    model = Sequential()
 
    #1st convolution layer
    model.add(Conv2D(16, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
    , padding='same', input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #2nd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #here compressed version

    #3rd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #4rd convolution layer
    model.add(Conv2D(16,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    model.summary()
    
    
    model.compile(optimizer=optimizer, loss=loss) #사용자 지정 파라미터(optimizer, loss)
    return model

def CNNAutoEncoder_96(optimizer, learning_rate, activation='sigmoid', loss='mse'):
    optimizer = optimizer_set(optimizer, learning_rate)
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(16, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
    , padding='same', input_shape=(96,96,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #2nd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #3rd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #here compressed version

    #4th convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #5th convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #6th convolution layer
    model.add(Conv2D(16,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation(activation))
    model.summary()
    
    #학습(약 7000번 정도 진행)
    model.compile(optimizer=optimizer, loss=loss) #사용자 지정 파라미터(optimizer, loss)
    return model