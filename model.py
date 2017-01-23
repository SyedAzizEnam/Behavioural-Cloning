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
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2YUV),(66,200))

def change_brightness(image):
    temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = 0.25 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def xy_translate(image, angle):
    x_translation = (100 * np.random.uniform()) - (100 / 2)
    new_angle = angle + ((x_translation / 100) * 2) * 0.1
    y_translation = (10 * np.random.uniform()) - (10 / 2)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0])), new_angle

def batch_generator(df, data_path, batch_size, img_rows, img_cols):

    num_rows = len(df.index)
    index = np.arange(1,num_rows)
    EPOCH = 1
    while 1:
        for j in range(num_rows//batch_size):
            np.random.shuffle(index)
            batch_files = df.loc[index[j*batch_size//3:(j+1)*batch_size//3]]
            train_images = np.zeros((batch_size, img_rows, img_cols, 3))
            train_steering = np.zeros(batch_size)
            for i,row in enumerate(batch_files.iterrows()):

                row = row[1]
                steering_angle = np.float32(row['steering'])

                center  = os.path.basename(row['center'])
                center = load_img(data_path+'IMG/'+center)
                center  = img_to_array(center)[60:140,:]

                left  = os.path.basename(row['left'])
                left = load_img(data_path+'IMG/'+left)
                left  = img_to_array(left)[60:140,:]

                right  = os.path.basename(row['right'])
                right = load_img(data_path+'IMG/'+right)
                right  = img_to_array(right)[60:140,:]

                flip = np.random.randint(2)
                if flip:
                    center  = np.fliplr(img_to_array(center))
                    left  = np.fliplr(img_to_array(left))
                    right  = np.fliplr(img_to_array(right))

                brightness_noise = np.random.randint(2)
                if brightness_noise:
                    center = change_brightness(center)
                    left = change_brightness(left)
                    right = change_brightness(right)

                translate = np.random.randint(2)
                if translate:
                    center,steering_angle = xy_translate(center,steering_angle)
                    left,__ = xy_translate(left,steering_angle)
                    right,__ = xy_translate(right,steering_angle)

                train_images[3*i], train_steering[3*i] = preprocess(center), -steering_angle if flip else steering_angle
                train_images[3*i+1], train_steering[3*i+1] = preprocess(left), -(steering_angle+0.15) if flip else steering_angle+0.15
                train_images[3*i+2], train_steering[3*i+2] = preprocess(right), -(steering_angle-0.15) if flip else steering_angle-0.15

            yield train_images,train_steering
        EPOCH += 1



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
train_data = batch_generator(df_train,data_path,batch_size=300,img_rows=img_rows,img_cols=img_cols)
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
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', name='FC2'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', name='FC3'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu', name='FC4'))
model.add(Dropout(0.2))
model.add(Dense(1, name='output'))
model.summary()
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss='mse', metrics=[])

model.fit_generator(train_data,
                    samples_per_epoch = 50000,
                    nb_epoch=10, verbose=1)

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
