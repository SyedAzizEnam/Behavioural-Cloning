import os
import numpy as np
import pandas as pd
from math import *
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, InputLayer, Lambda, Dropout
from keras.layers import Convolution2D,Flatten
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.models import model_from_json

def preprocess(image):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[50:150,40:280], (66,200))

def batch_generator(df, data_path, batch_size, img_rows, img_cols):
    num_rows = len(df.index)

    while 1:
        index = np.random.randint(1,num_rows-1, batch_size//3)
        batch_files = df.loc[index]
        train_images = np.zeros((batch_size, img_rows, img_cols, 3))
        train_steering = np.zeros(batch_size)
        for i, row in enumerate(batch_files.iterrows()):

            center  = os.path.basename(row[1]['center'])
            left = os.path.basename(row[1]['left'])
            right = os.path.basename(row[1]['right'])

            center = load_img(data_path+'IMG/'+center)
            left = load_img(data_path+'IMG/'+left)
            right = load_img(data_path+'IMG/'+right)

            center  = preprocess(cv2.cvtColor(img_to_array(center), cv2.COLOR_RGB2YUV))
            left = preprocess(cv2.cvtColor(img_to_array(left), cv2.COLOR_RGB2YUV))
            right = preprocess(cv2.cvtColor(img_to_array(right), cv2.COLOR_RGB2YUV))

            train_images[3*i], train_steering[3*i] = center, np.float32(row[1]['steering'])
            train_images[3*i+1], train_steering[3*i+1] = left, np.float32(row[1]['steering'])+0.15
            train_images[3*i+2], train_steering[3*i+2] = right, np.float32(row[1]['steering'])-0.15

        yield train_images, train_steering

img_rows, img_cols = 200, 66
data_path = './train_data/'
val_data_path = './val-data/'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df_train = pd.read_csv(data_path+'driving_log.csv', names=columns)
df_train.drop(df_train.index[[0]], inplace=True)
df_train.dropna(how='any', axis=0, inplace=True)
"""
df_val = pd.read_csv(val_data_path+'driving_log.csv', names=columns)
df_val.drop(df_val.index[[0]], inplace=True)
df_val.dropna(how='any', axis=0, inplace=True)
"""
train_data = batch_generator(df_train,data_path,batch_size=512,img_rows=img_rows,img_cols=img_cols)
#val_data = batch_generator(df_val,val_data_path,batch_size=1024,img_rows=img_rows,img_cols=img_cols)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols, 3), name='Normalization'))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', name='Conv1'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', name='Conv2'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', name='Conv3'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv4'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv5'))
model.add(Flatten())
model.add(Dense(1164, activation='relu', name='FC1'))
model.add(Dense(100, activation='relu', name='FC2'))
model.add(Dense(50, activation='relu', name='FC3'))
model.add(Dense(10, activation='relu', name='FC4'))
model.add(Dense(1, name='output'))
model.summary()
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])

model.fit_generator(train_data,
                    samples_per_epoch = 512*50,
                    nb_epoch=4, verbose=1)

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
