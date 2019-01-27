from keras import applications as pretrained
from keras.models import Model
from keras.preprocessing import image
from keras import optimizers
import numpy as np
import PIL
from progress.bar import Bar
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from DataGenerator import DataGenerator


# X_train = np.load('/var/tmp/pksm/self_driving_data/data/train_data.npy')

# train = np.load('/var/tmp/pksm/self_driving_data/data/train_data.npy')

# y_train = np.load('/var/tmp/pksm/self_driving_data/data/train_labels.npy')

# # X_valid = np.load('/var/tmp/pksm/self_driving_data/data/valid_data.npy')
# y_valid = np.load('/var/tmp/pksm/self_driving_data/data/valid_labels.npy')

data_gen = DataGenerator('/var/tmp/pksm/self_driving_data/data/')
X_train, y_train = data_gen.load_data(usage='train')
X_train, y_train = data_gen.preprocess_data(X_train, y_train, balance='undersampling')

X_valid, y_valid = data_gen.load_data(usage='valid')
X_valid, y_valid = data_gen.preprocess_data(X_valid, y_valid, balance='undersampling')



##################

batch_size = 256
epochs = 5

model = pretrained.mobilenet.MobileNet(weights=None, classes=3)

sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# y_train_hot = np_utils.to_categorical(y_train, 3)
# y_valid_hot = np_utils.to_categorical(y_valid, 3)

# X_train = X_train.reshape((-1, 224, 224, 3))
# X_valid = X_valid.reshape((-1, 224, 224, 3))

stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1)

model.fit(X_train, 
          y_train_hot, 
          validation_data=(X_valid, y_valid), 
          epochs=epochs, 
          batch_size=batch_size,
          callbacks = [stopper],
          shuffle=True,
          verbose=1)

# model.fit_generator(generator(X_train, y_train, batch_size),
#                     validation_data=generator(X_valid, y_valid, batch_size),
#                     validation_steps=np.ceil(X_valid.shape[0] / batch_size).astype(int),
#                     samples_per_epoch = np.ceil(X_train.shape[0] / batch_size).astype(int),
#                     epochs=epochs, 
#                     verbose=2)

# model.fit(X_train, 
#           y_train, 
#           validation_data=(X_valid, y_valid), 
#           epochs=epochs, 
#           batch_size=batch_size,
#           verbose=2)

model.save("self-driving-data-mobilenet.h5")
print("Saved model to disk")
score = model.evaluate(X_valid, y_valid, verbose=0)
print('Valid loss:', score[0])
print('Valid accuracy:', score[1])

