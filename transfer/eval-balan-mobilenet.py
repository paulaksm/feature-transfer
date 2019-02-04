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
from progress.bar import Bar
from keras.utils import to_categorical
from keras.utils import np_utils
from DataGenerator import DataGenerator

shape = (45, 80, 3)
target_size = (224, 224)
flat_shape = target_size[0] * target_size[1] * 3

def preprocessing(dataset):
    all_images = []
    print('Transforming images...')
    bar = Bar('Transforming for MobileNet standard', max=dataset.shape[0])
    for img in dataset:
        img = img.reshape(shape)
        img = image.array_to_img(img, data_format='channels_last')
        img = img.resize(target_size, resample=PIL.Image.NEAREST)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = pretrained.mobilenet.preprocess_input(img)
        img = img.reshape(flat_shape)
        all_images.append(img)
        bar.next()    
    bar.finish()
    processed_data = np.array(all_images)
    return processed_data

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
                                                        balance='undersampling')
    del data
    model = load_model(user_args.trained_model)
    evaluate(model, prep_data, prep_labels)


if __name__ == '__main__':
    main()
