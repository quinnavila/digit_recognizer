
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def prepare_data(df):
    X = df.drop(['label'],axis=1)
    y = df.label
    # normalize data and reshape
    X = X.values / 255.0
    X = X.reshape(-1, 28, 28, 1)

    return X, y

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)

    model=Sequential()

    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    

    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
        
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
        
    model.add(Dense(10,activation="softmax"))
        
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

    score = model.evaluate(X_val, y_val, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return model


def main_training():
    df = pd.read_csv('data/train.csv')

    X, y = prepare_data(df)

    model = train_model(X, y)

    model_filename = 'digits_model1.h5'
    model.save(model_filename)
    print(f'Model saved: {model_filename}')
    

if __name__=="__main__":
    main_training()