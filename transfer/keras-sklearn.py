import os
import shutil
import json
import datetime
import numpy as np
from keras import applications as pretrained
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Conv2D, Reshape, Activation, Dense
from keras.initializers import VarianceScaling
from DataGenerator import DataGenerator
from sklearn.neural_network import MLPClassifier

###############################################

model = pretrained.mobilenet.MobileNet(weights='imagenet')
for l in model.layers:
  l.trainable = False
model2 = Model(inputs=model.input, outputs=model.get_layer('dropout').output)

data_gen = DataGenerator('/var/tmp/pksm/self_driving_data/data')
x_train, y_train = data_gen.load_data(usage='train')
x_valid, y_valid = data_gen.load_data(usage='valid')

x_train, y_train = data_gen.preprocess_data(x_train, y_train, balance='equals')
x_valid, y_valid = data_gen.preprocess_data(x_valid, y_valid, balance='equals')

x_mob_train = None
x_mob_valid = None

for i in x_train:
    i = np.expand_dims(i, axis=0)
    emb = model.predict(i)
    emb = emb.flatten()
    if x_mob_train is None:
        x_mob_train = np.array([], dtype=x_train[0].dtype).reshape(0, emb.shape[0])
    x_mob_train = np.concatenate((x_mob_train, emb), axis=0)

for i in x_valid:
    i = np.expand_dims(i, axis=0)
    emb = model.predict(i)
    emb = emb.flatten()
    if x_mob_valid is None:
        x_mob_valid = np.array([], dtype=x_valid[0].dtype).reshape(0, emb.shape[0])
    x_mob_valid = np.concatenate((x_mob_valid, emb), axis=0)

y_train = np.ravel(y_train)
y_valid = np.ravel(y_valid)

nn = MLPClassifier(alpha=0.0001,  
                   verbose=1, 
                   solver='adam',
                   activation='relu',
                   max_iter=200,
                   early_stopping=True,
                   hidden_layer_sizes=(200,))

print("Fit..")
nn.fit(x_mob_train, y_train)
print("Predict..")
predictions = nn.predict(x_mob_valid)

score = nn.score(x_mob_valid, y_valid)
print(score)
