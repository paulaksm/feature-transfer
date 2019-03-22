'''
First attempt to iterative pruning method proposed by Han 2015
'''
import os
import numpy as np
import argparse
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from DataGenerator import DataGenerator
from keras.layers import Dense, Flatten, Input
import keras.backend as K


class CustomCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        # zero out weights accordingly to prune_mask list 
        final = []
        iter_weights = self.model.get_weights()
        for i in range(len(iter_weights)):
            final.append(np.multiply(iter_weights[i], prune_mask[i]))
        self.model.set_weights(final)

    def on_train_end(self, epoch, logs=None):
        test_data, test_labels = data_gen.load_data(usage='test')
        test_data, test_labels = data_gen.preprocess_data(test_data, 
                                                           test_labels,
                                                           balance='equals',
                                                           raw=True)
        print('Evaluating model on {} samples'.format(test_labels.shape[0]))
        print('Class distribution:')
        for i in range(3):
            print('{} : {}'.format(i, np.sum(test_labels[:, i]).astype(int)))
        scores = self.model.evaluate(test_data, test_labels, verbose=1)
        print("Model performance on test dataset: {}".format(scores[1]))

def parse():
    description = 'Iterative pruning - v0'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-data_path',
                        '--data',
                        type=str, help='path to data folder')
    parser.add_argument('-base_model',
                        '--model',
                        type=str, help='path to base model file')
    parser.add_argument('-keep_iter',
                        '--keep',
                        type=float, default=0.7,
                        help='percentage to keep per iteration (default=0.7)')
    parser.add_argument('-iter',
                        '--iter',
                        type=int, default=5,
                        help='number of iterations (default=5)')

    return parser.parse_args()



def iterative(path_data, model, keep_ratio, iterations):
    pr = 1
    model = load_model(model)
    for i in range(iterations):
        print("Pruning and retraining: {} iteration".format(i))
        pr = pr * keep_ratio
        all_weights = model.get_weights()
        list_flat = []
        for j in range(len(all_weights)):
          list_flat.append(all_weights[j].reshape(-1))
        flat = np.concatenate(list_flat)
        sort_flat = sorted(map(abs, flat))
        threshold = sort_flat[int(len(sort_flat) * (1-pr))]
        del list_flat, sort_flat
        print("Ratio of weights kept: {}".format(pr))
        nonzero = np.count_nonzero(flat)
        print("Nonzero parameters before pruning: {}".format(nonzero))
        print("Total parameters: {}".format(flat.shape[0]))
        print("Parameter compression rate: {}x".format(int(flat.shape[0]/nonzero)))
        global prune_mask
        prune_mask = []
        curr_weights = []
        for i in range(len(all_weights)):
            prune_mask.append(abs(all_weights[i]) > threshold)
            curr_weights.append(np.multiply(all_weights[i], prune_mask[i]))
        train(path_data, curr_weights, pr)



def train(dataset_path, custom_weights, pr):
    # path_checkpoints = '/content/gdrive/Team Drives/Models/best_model.hdf5'
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

    x = Dense(1333, activation='relu')(inputs)
    x = Dense(200, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.set_weights(custom_weights) # initializing weights from first training + pruned connections

    sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=0)

    # checkpoint = ModelCheckpoint(path_checkpoints,
    #                              monitor='val_acc',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='max')

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
                        callbacks=[stopper, custom_callback],
                        epochs=epochs,
                        verbose=0)
    # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # print("Number of nonzero trainable parameters: {}".format(np.nonzero(trainable_count)))
    model.save('/content/gdrive/Team Drives/Models/weight-kept-ratio-{}.hdf5'.format(pr))


def main():
    args = parse()
    iterative(args.data, args.model, args.keep, args.iter)


if __name__ == '__main__':
    main()
