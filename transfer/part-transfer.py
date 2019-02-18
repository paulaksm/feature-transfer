'''
Script for experiment AnB

'''
import os
import numpy as np
import argparse
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from DataGenerator import DataGenerator
from keras import applications as pretrained
from keras.layers import GlobalAveragePooling2D, Conv2D, Activation, Reshape, Dropout
from keras.backend import int_shape
from keras.initializers import VarianceScaling


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass
        # layer = self.model.get_layer('conv_pw_1')
        # weights = layer.get_weights()
        # msg = "Weights changed after epoch {}".format(epoch+1)
        # assert np.array_equal(frozen_weights, weights), msg


def parse():
    description = 'Transfer network'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-data_path',
                        '--data',
                        type=str, help='path to data folder')
    parser.add_argument('-freeze',
                        '--n_freeze',
                        type=int, help='number of blocks to freeze')

    return parser.parse_args()


def keep(model,
         base):
    first = ['conv1_pad',
             'conv1',
             'conv1_bn',
             'conv1_relu']
    for l in first:
        layer = model.get_layer(l)
        base_layer = base.get_layer(l)
        # layer.trainable = False  
        layer.set_weights(base_layer.get_weights())


def frozen(model,
           base,
           n=1):
    dsc = ['conv_dw_{}', 
           'conv_dw_{}_bn', 
           'conv_dw_{}_relu', 
           'conv_pw_{}',
           'conv_pw_{}_bn',
           'conv_pw_{}_relu']
    for i in range(1, n+1):
        for layer_name in dsc:
            layer = model.get_layer(layer_name.format(i))
            base_layer = base.get_layer(layer_name.format(i))
            # layer.trainable = False  
            layer.set_weights(base_layer.get_weights())


def train(dataset_path, freeze_n):
    path_checkpoints = '/content/gdrive/Team Drives/Models/model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    file_train = os.path.join(dataset_path, 'train_labels.npy')
    file_valid = os.path.join(dataset_path, 'valid_labels.npy')
    x_train = np.load(file_train, mmap_mode='r')
    x_train_samples = x_train.shape[0]

    x_valid = np.load(file_valid, mmap_mode='r')
    x_valid_samples = x_valid.shape[0]

    del x_train, x_valid

    batch_size = 64
    epochs = 15
    base_model = pretrained.mobilenet.MobileNet(weights='imagenet') # Task A model
    model = pretrained.mobilenet.MobileNet(weights=None, classes=3)

    # global frozen_weights
    # frozen_weights = base_model.get_layer('conv_pw_1').get_weights()
    keep(model, base_model)
    frozen(model, base_model, n=freeze_n)
    
    del base_model

    # model = pretrained.mobilenet.MobileNet(weights='imagenet')
    model2 = Model(inputs=model.input, outputs=model.get_layer('conv_pw_{}_relu'.format(freeze_n)).output)
    glob_pool = GlobalAveragePooling2D()(model2.output)
    res_1 = Reshape((1, 1, int_shape(glob_pool)[1]), name='pool_reshape')(glob_pool)
    drop = Dropout(0.001)(res_1)
    conv2d = Conv2D(3,
                   (1, 1), 
                   padding='same', 
                   data_format='channels_last', 
                   kernel_initializer=VarianceScaling(distribution='uniform', mode='fan_avg'))(drop)
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
    
    custom_callback = CustomCallback()

    data_gen = DataGenerator(dataset_path)

    model.fit_generator(data_gen.npy_generator(usage='train', batch_size=batch_size),
                        steps_per_epoch=np.ceil(x_train_samples / batch_size).astype(int),
                        validation_data=data_gen.npy_generator(usage='valid', batch_size=batch_size),
                        validation_steps=np.ceil(x_valid_samples / batch_size).astype(int),
                        callbacks=[stopper, checkpoint, custom_callback],
                        epochs=epochs,
                        verbose=1)

    model.save("A{}B-little.h5".format(freeze_n))
    print("Saved model to disk")


def main():
    args = parse()
    train(args.data, args.n_freeze)


if __name__ == '__main__':
    main()
