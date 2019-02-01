import os
import shutil
import datetime
import numpy as np
import argparse
from keras import applications as pretrained
from keras.models import Model, load_model
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from DataGenerator import DataGenerator

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer('conv_pw_1')
        weights = layer.get_weights()
        assert np.array_equal(frozen_weights, weights), "Weights change after epoch {}".format(epoch)

def parse():
    description = 'Selffer network'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-data_path',
                        '--data',
                        type=str, help='path to data folder')
    parser.add_argument('-freeze',
                        '--n_freeze',
                        type=int, help='number of blocks to freeze')
    parser.add_argument('-base',
                        '--base',
                        type=str, help='path to base model')

    return parser.parse_args()


def npy_generator(dataset_path, usage='train', batch_size=64):
    data_gen = DataGenerator(dataset_path)
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

def frozen(model,
           n=1):
    dsc = [ 'conv_dw_{}', 
            'conv_dw_{}_bn', 
            'conv_dw_{}_relu', 
            'conv_pw_{}',
            'conv_pw_{}_bn',
            'conv_dw_{}_relu' ]
    for i in range(1, n+1):
        for layer in dsc:
            model.get_layer(layer.format(i)).trainable = False  


def train(dataset_path, base_model_path, freeze_n):
    path_checkpoints = 'model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    file_data = os.path.join(dataset_path, '_data.npy')
    file_label = os.path.join(dataset_path, '_labels.npy')
    batch_size = 64
    epochs = 15
    model = load_model(base_model_path)
    frozen_weights = model.get_layer('conv_pw_1').get_weights()
    frozen(model, n=freeze_n)
    
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
    custom_callback = MyCallback()

    hist = model.fit_generator(npy_generator(usage='train', batch_size=batch_size),
                               steps_per_epoch = np.ceil(x_train_samples / batch_size).astype(int),
                               validation_data=npy_generator(usage='valid', batch_size=batch_size),
                               validation_steps=np.ceil(x_valid_samples / batch_size).astype(int),
                               callbacks = [stopper, checkpoint, custom_callback],
                               epochs=epochs, 
                               verbose=1)

    model.save("B{}B-mobilenet.h5".format(freeze_n))
    print("Saved model to disk")


def main():
    global frozen_weights
    args = parse()
    train(args.data, args.base, args.n_freeze)


if __name__ == '__main__':
    main()