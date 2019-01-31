import os
import shutil
import json
import datetime
import numpy as np
from keras import applications as pretrained
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Reshape, Activation
from keras.initializers import VarianceScaling
from DataGenerator import DataGenerator

global data_gen
data_gen = DataGenerator('/content/data/')

def npy_generator(dataset_path='/content/data/', usage='train', batch_size=64):
    file = os.path.join(dataset_path, usage)
    file_data = file + '_data.npy'
    file_label = file + '_labels.npy'
    x = np.load(file_data, mmap_mode='r')
    y = np.load(file_label, mmap_mode='r')
    while True:
        init_idx = 0
        end_idx = batch_size
        for i in range(np.ceil(x.shape[0]/batch_size).astype(int)):
            x_batch = x[init_idx:end_idx][:]
            y_batch = y[init_idx:end_idx]
            x_batch, y_batch = data_gen.preprocess_data(x_batch, y_batch)
            init_idx += batch_size
            end_idx += batch_size
            if end_idx > x.shape[0]:
                end_idx = x.shape[0]
            yield x_batch, y_batch

start_time = datetime.datetime.now()

cwd = os.getcwd()
dir_checkpoints = os.path.join(cwd, 'checkpoints')
if os.path.exists(dir_checkpoints):
    shutil.rmtree(dir_checkpoints)
os.makedirs(dir_checkpoints)
d = os.path.expanduser(dir_checkpoints)
path_checkpoints = os.path.join(d, 'model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')

x_train = np.load('/content/data/train_data.npy', mmap_mode='r')
x_train_samples = x_train.shape[0]

x_valid = np.load('/content/data/valid_data.npy', mmap_mode='r')
x_valid_samples = x_valid.shape[0]

del x_train, x_valid

###############################################

batch_size = 64
epochs = 15

model = pretrained.mobilenet.MobileNet(weights='imagenet')
model2 = Model(inputs=model.input, outputs=model.get_layer('dropout').output)
conv2d = Conv2D(3,
               (1, 1), 
               padding='same', 
               data_format='channels_last', 
               kernel_initializer=VarianceScaling(distribution='uniform', mode='fan_avg'))(model2.output)
act = Activation('softmax')(conv2d)
res = Reshape((3,), name='last_reshape')(act)

model = Model(inputs=model2.input, outputs=res)


sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1)

checkpoint = ModelCheckpoint(path_checkpoints, 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

hist = model.fit_generator(npy_generator(usage='train', batch_size=batch_size),
                           steps_per_epoch = np.ceil(x_train_samples / batch_size).astype(int),
                           validation_data=npy_generator(usage='valid', batch_size=batch_size),
                           validation_steps=np.ceil(x_valid_samples / batch_size).astype(int),
                           callbacks = [stopper, checkpoint],
                           epochs=epochs, 
                           verbose=1)


model.save("self-driving-data-mobilenet.h5")
print("Saved model to disk")

file_name = 'hist_{}{}{}_{}{}.json'.format(start_time.year, 
                                           start_time.month, 
                                           start_time.day, 
                                           start_time.hour, 
                                           start_time.minute)
with open(file_name, 'w') as file:
  json.dump(hist.history, file)


