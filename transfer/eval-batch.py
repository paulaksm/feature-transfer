'''
 Script for evaluating MobileNet with (balanced) testing dataset (self-driving car data)

 Improvements:
 - passar modelos em lista de uma vez
'''
import PIL
import os
import argparse
import numpy as np
from keras import applications as pretrained
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras.utils import np_utils
from DataGenerator import DataGenerator

shape = (45, 80, 3)
target_size = (224, 224)
flat_shape = target_size[0] * target_size[1] * 3



def main():
    """
    Script for evaluating MobileNet with testing dataset (self-driving car data)
    """
    description = "Evaluate MobileNet with test dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-model',
                        '--trained_model',
                        type=str, help='MobileNet trained model')
    parser.add_argument("-data_path",
                        "--data",
                        type=str,
                        help="Path to dataset directory")

    user_args = parser.parse_args()
    batch_size = 16
    path = os.path.join(user_args.data, 'test_labels.npy')
    labels = np.load(path)
    data_gen  = DataGenerator(user_args.data)
    
    print('Evaluating model on {} samples'.format(labels.shape[0]))
    print('Class distribution:')
    count = np.unique(labels, return_counts=True)
    for i in range(3):
        print('{} : {}'.format(count[0][i], count[1][i]))
    model = load_model(user_args.trained_model)
    scores = model.evaluate_generator(data_gen.npy_generator(usage='test', batch_size=batch_size),
                                      steps=np.ceil(labels.shape[0] / batch_size).astype(int),
                                      verbose=1)
    print("Model performance: {}".format(scores[1]))


if __name__ == '__main__':
    main()
