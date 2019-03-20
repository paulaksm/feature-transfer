'''
Script for training DFN on self-driving data...
TODO: iterative pruning method proposed by Han 2015
'''
import os
import numpy as np
import argparse
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from DataGenerator import DataGenerator
from keras.layers import Dense, Flatten, Input


class CustomCallback(Callback):
    def on_train_end(self, epoch, logs=None):
        model = load_model("/content/gdrive/Team Drives/Models/best_model.hdf5")
        test_data, test_labels = data_gen.load_data(usage='test')
        test_data, test_labels = data_gen.preprocess_data(test_data, 
                                                           test_labels,
                                                           balance='equals',
                                                           raw=True)
        print('Evaluating model on {} samples'.format(test_labels.shape[0]))
        print('Class distribution:')
        for i in range(3):
            print('{} : {}'.format(i, np.sum(test_labels[:, i]).astype(int)))
        scores = model.evaluate(test_data, test_labels, verbose=1)
        print("Best model performance on test dataset: {}".format(scores[1]))


def parse():
    description = 'DFN self-driving car'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-data_path',
                        '--data',
                        type=str, help='path to data folder')
    return parser.parse_args()


def train(dataset_path):
    path_checkpoints = '/content/gdrive/Team Drives/Models/best_model.hdf5'
    # path_checkpoints = '/content/gdrive/Team Drives/Models/model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    file_train = os.path.join(dataset_path, 'train_labels.npy')
    file_valid = os.path.join(dataset_path, 'valid_labels.npy')
    x_train = np.load(file_train, mmap_mode='r')
    x_train_samples = x_train.shape[0]

    x_valid = np.load(file_valid, mmap_mode='r')
    x_valid_samples = x_valid.shape[0]

    del x_train, x_valid

    batch_size = 64
    epochs = 25

    # Model
    inputs = Input(shape=(10800,)) #working with already flatten image

    x = Dense(276, activation='relu')(inputs)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

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

    global data_gen
    data_gen = DataGenerator(dataset_path)


    model.fit_generator(data_gen.npy_generator(usage='train', batch_size=batch_size),
                        steps_per_epoch=np.ceil(
                            x_train_samples / batch_size).astype(int),
                        validation_data=data_gen.npy_generator(
                            usage='valid', batch_size=batch_size),
                        validation_steps=np.ceil(
                            x_valid_samples / batch_size).astype(int),
                        callbacks=[stopper, checkpoint, custom_callback],
                        epochs=epochs,
                        verbose=1)


def main():
    args = parse()
    train(args.data)


if __name__ == '__main__':
    main()
