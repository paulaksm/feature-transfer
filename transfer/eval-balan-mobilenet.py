'''
 Script for evaluating MobileNet with (balanced) testing dataset (self-driving car data)

 Improvements:
 - passar modelos em lista de uma vez
'''
import PIL
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

def evaluate(model, data, labels):
    scores = model.evaluate(data, labels, verbose=1)
    print("Model performance: {}".format(scores[1]))
    return


def main():
    """
    Script for evaluating MobileNet with testing dataset (self-driving car data)
    """
    description = "Evaluate MobileNet with test dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-model',
                        '--trained_model',
                        type=str, help='MobileNet trained model')
    parser.add_argument("-data",
                        "--data",
                        type=str,
                        help="Path to dataset Images")
    parser.add_argument("-labels",
                        "--labels",
                        type=str,
                        help="Path to dataset Labels")

    user_args = parser.parse_args()

    data = np.load(user_args.data)
    labels = np.load(user_args.labels)
    data_gen  = DataGenerator('')
    prep_data, prep_labels  = data_gen.preprocess_data(data, 
                                                       labels,
                                                       balance='equals')
    del data
    print('Evaluating model on {} samples'.format(prep_labels.shape[0]))
    print('Class distribution:')
    for i in range(3):
        print('{} : {}'.format(i, np.sum(prep_labels[:, i]).astype(int)))
    model = load_model(user_args.trained_model)
    evaluate(model, prep_data, prep_labels)


if __name__ == '__main__':
    main()
