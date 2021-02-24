from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D


def autoencoder_model(input_shape=(28,28,1), optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()
    
    #1st convolution layer
    #16 is number of filters and (3, 3) is the size of the filter
    model.add(Conv2D(16, (3, 3) , padding='same', input_shape=input_shape))
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

    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    return model