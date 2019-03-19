import os
import PIL
import numpy as np
# from progress.bar import Bar
from keras import applications as pretrained
from keras.preprocessing import image
from keras.utils import to_categorical


class DataGenerator:
    """
    A class to handle the image preprocessing required for
    MobileNet architecture.
    :param dataset_path: path to self-driving dataset folder
    :type dataset_path: str
    :param source_shape: input shape of the images to be transformed
    :type source_shape: tuple of ints in 'channels_last' format
    """

    def __init__(self,
                 dataset_path,
                 source_shape=(45, 80),
                 target_model=pretrained.mobilenet):
        self.path = dataset_path
        self.shape = source_shape
        self.target_model = target_model

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, user_path):
        self._path = os.path.expanduser(user_path)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def target_model(self):
        return self._target_model

    @target_model.setter
    def target_model(self, value):
        self._target_model = value

    def _undersampling(self,
                       data,
                       labels):
        _, count = np.unique(labels,
                             return_counts=True)
        majority_class = np.argmax(count)
        count.sort()
        size_2nd_highest = count[-2]  # second highest in count variable
        idx_up = np.argwhere(labels == majority_class)
        remove_size = idx_up.shape[0] - size_2nd_highest
        idx_sel = np.random.choice(idx_up.flatten(),
                                   size=remove_size,
                                   replace=False)
        all_idx = np.array(range(0, labels.shape[0]))
        idx_sel = np.delete(all_idx, idx_sel)
        data = data[idx_sel]
        labels = labels[idx_sel]
        return data, labels

    def _equals(self,
                data,
                labels):
        _, count = np.unique(labels,
                             return_counts=True)
        minority_class = np.argmin(count)
        size_minor = count[minority_class]
        all_idx = np.array(range(0, labels.shape[0]))
        result = np.array([], dtype=np.int)
        for arg, i in enumerate(count):
            if arg == minority_class:
                continue
            idx = np.argwhere(labels == arg)
            remove_size = i - size_minor
            idx_rem = np.random.choice(idx.flatten(),
                                       size=remove_size,
                                       replace=False)
            result = np.append(result, idx_rem)
        idx_sel = np.delete(all_idx, result)
        data = data[idx_sel]
        labels = labels[idx_sel]
        return data, labels

    def load_data(self,
                  usage='train'):
        """
        Load ndarray from npy files
        :param type: type of dataset to be loaded (train, valid or test)
        :type type: str
        :return: data, labels
        :rtype: ndarray (shape=(N, flatten_input_shape)), ndarray (shape=(N,))
        """
        file = os.path.join(self.path, usage)
        file_data = file + '_data.npy'
        file_label = file + '_labels.npy'
        msg = '{} not found. Please check dataset_path instance variable\
            and usage variable.'
        assert os.path.isfile(file_data), msg.format('Data')
        assert os.path.isfile(file_label), msg.format('Labels')
        data = np.load(file_data)
        labels = np.load(file_label)
        assert data.shape[0] == labels.shape[0], 'Shape mismatch.'
        assert data.shape[1] == self.shape[0] * self.shape[1] * 3,\
            'DataGenerator shape {} does not match data shape {}'\
            .format(self.shape, data.shape)
        return data, labels

    def preprocess_data(self,
                        data,
                        labels,
                        target_shape=(224, 224),
                        balance=None):
        """
        Preprocess data to be compliant with input_shape for a specific CNN
        architecture. Default is MobileNet v1.
        :param data: array of flatten images
        :type data: ndarray (shape=(N, flatten_shape))
        :param labels: array of image labels
        :type labels: ndarray (shape=(N,))
        :param target_shape: desired shape after preprocess
        :type target_shape: tuple of ints. Default is (224,224).
        :param balance: balance data by undersampling if 'undersampling'.\
            Default is None.
        :type balance: str
        :param target_model: preprocess specific to a target_model.\
            Default is keras.applications.mobilenet.
        :type target_model: Keras module
        :return: data, labels
        :rtype: ndarray (shape=(N, 224, 224, 3)), ndarray (shape=(N, 3))
        """
        balance_methods = [None, 'undersampling', 'equals']
        assert balance in balance_methods, \
                "Not supported method: {}".format(balance)
        if balance == 'undersampling':
            data, labels = self._undersampling(data, labels)
        if balance == 'equals':
            data, labels = self._equals(data, labels)
        shape = (self.shape[0], self.shape[1], 3)
        all_images = []
        flat_shape = target_shape[0] * target_shape[1] * 3
        for img in data:
            img = img.reshape(shape)
            img = image.array_to_img(img, data_format='channels_last')
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = self.target_model.preprocess_input(img)
            img = img.reshape(flat_shape)
            all_images.append(img)
        data = np.array(all_images)
        data = data.reshape((-1, target_shape[0], target_shape[1], 3))
        labels = to_categorical(labels, 3)
        assert data.shape[0] == labels.shape[0], 'Shape mismatch.'
        return data, labels

    def npy_generator(self,
                      usage='train',
                      preprocess=False,
                      batch_size=64):
        '''
        Generator used in Keras
        
        '''
        file = os.path.join(self.path, usage)
        file_data = file + '_data.npy'
        file_label = file + '_labels.npy'
        x = np.load(file_data, mmap_mode='r')
        y = np.load(file_label, mmap_mode='r')
        while True:
            init_idx = 0
            end_idx = batch_size
            permutation = np.random.permutation(x.shape[0])
            for i in range(np.ceil(x.shape[0] / batch_size).astype(int)):
                sel_idx = permutation[init_idx:end_idx]
                x_batch = x[sel_idx]
                y_batch = y[sel_idx]
                if preprocess:
                    x_batch, y_batch = self.preprocess_data(x_batch, y_batch)
                else:
                    x_batch = x_batch.copy()
                    x_batch = x_batch.astype('float32') / 255
                    y_batch = to_categorical(y_batch, 3)
                init_idx += batch_size
                end_idx += batch_size
                if end_idx > x.shape[0]:
                    end_idx = x.shape[0]
                yield x_batch, y_batch

# def main():
#     x = DataGenerator('~/self_driving_data/data/')
#     dt, lsd = x.load_data(usage='valid')
#     _, _ = x.preprocess_data(dt, lsd)


# if __name__ == '__main__':
#     main()
