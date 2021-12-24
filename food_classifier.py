"""
Project:    The idea is to implement a convolutional neural network which is able to identify and separate different classes:
            for instance FOOD and NON-FOOD.
"""
from datetime import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow import expand_dims
import os
import numpy as np


def cnn_classifier(directory, dir_proj, h, w, d, BS = 8):

    t0 = time()
    image_size = (h, w)  # image size (resize) can affect the accuracy !
    input_shape = (h, w, d)
    model_name = 'model_1e-3.h5'

    """
        Loading the images:
        - Please download all the images and put them into a directory with two sub-directory: FOOD and NON-FOOD
        - The user may select the right directory on his laptop
    """

    # create data-sets
    dataset_train = image_dataset_from_directory(directory, labels='inferred', label_mode='int',
                                                 class_names=None, color_mode='rgb', batch_size=BS, image_size=image_size,
                                                 shuffle=True,
                                                 seed=123, validation_split=0.2, subset='training',
                                                 interpolation='bilinear', follow_links=False,
                                                 crop_to_aspect_ratio=False)

    dataset_test = image_dataset_from_directory(directory, labels='inferred', label_mode='int',
                                                class_names=None, color_mode='rgb', batch_size=BS, image_size=image_size,
                                                shuffle=False,
                                                seed=123, validation_split=0.2, subset='validation',
                                                interpolation='bilinear', follow_links=False,
                                                crop_to_aspect_ratio=False)

    class_names = dataset_train.class_names
    print('Existing classes: ', class_names)

    for image_batch, labels_batch in dataset_train:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # creation of a classifier - Neural network with some layers to train the data-sets
    classifier = Sequential()
    # classifier.add(RandomFlip("horizontal", input_shape=input_shape))  # data augmentation
    # classifier.add(RandomRotation(0.1))  # data augmentation
    # classifier.add(RandomZoom(0.1))  # data augmentation
    classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Rescaling(1. / 255))  # rescaling, Normalize pixel values to be between 0 and 1
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(Flatten())
    #classifier.add(Dropout(0.2))  # data augmentation
    classifier.add(Dense(64, activation='sigmoid'))
    classifier.add(Dense(1, activation='sigmoid'))  # only two values [0 1]
    classifier.summary()

    opt = Adam(learning_rate=1e-3)  # step of optimisation (0.001 default)

    # compile the model
    classifier.compile(optimizer=opt,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    # you can use callbacks to save your model and load the model parameter at a later stage
    checkpoint = ModelCheckpoint(model_name,
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1,
                                 mode='max')

    auto_stop = EarlyStopping(monitor='val_accuracy',
                              mode='max',
                              patience=10)  # check the 10 last epoch values

    # train the model
    history = classifier.fit(dataset_train,
                             validation_data=dataset_test,
                             batch_size=BS,
                             epochs=1000,  # increase to avoid any previous stop
                             callbacks=[checkpoint, auto_stop])

    # restore best weights
    classifier.load_weights(model_name)

    # evaluation
    plt.figure(1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('accuracy.png')

    plt.figure(2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig('loss.png')
    plt.show()

    test_loss, test_acc = classifier.evaluate(dataset_test, verbose=2)
    print(test_acc)
    print('Done in %0.3fs' % (time() - t0))

    return test_acc, (time() - t0)


def to_predict(dir_new, saved_model, h, w,):

    image_size = (h, w)  # image size (resize) can affect the accuracy !
    class_names = ['FOOD', 'NONFOOD']
    nonfood_cnt = 0
    food_cnt = 0
    conf_int = [0.4, 0.6]
    unknown = 0
    pred_array = []

    path, dirs, files = next(os.walk(dir_new))
    file_count = len(files)

    # Load the model
    model = load_model(saved_model)

    for image_nr in range(1, file_count+1):
        # classify new pictures (not given neither during training and validation) using the model
        img_name = 'image ' + '(' + str(image_nr) + ').jpg'
        img = load_img(dir_new + img_name, target_size=image_size)
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        if predictions[0] > conf_int[1]:
            nonfood_cnt = nonfood_cnt + 1
        elif predictions[0] < conf_int[0]:
            food_cnt = food_cnt + 1
        else:
            unknown = unknown + 1
        print(img_name, "most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_names[0], 100 - 100 * np.max(predictions[0])))

    pred_array.append(100 - 100 * np.max(predictions[0]))  # for histogram (100 - 100 * np.max(predictions[0]) -> FOOD

    plt.hist(pred_array, 10, facecolor='blue', alpha=0.5)
    plt.title("Histogram:" + class_names[0])  # change the class name for the histogram
    plt.xlabel('confidence')
    plt.ylabel('samples')
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.savefig('histogram_food.png')

    plt.show()

    print("The number of images associated to FOOD are "
          + str(food_cnt) + " out of " + str(file_count)
          + " ---> " + str(food_cnt/file_count * 100) + " %")
    print("The number of images associated to NON_FOOD are "
          + str(nonfood_cnt) + " out of " + str(file_count)
          + " ---> " + str(nonfood_cnt/file_count * 100) + " %")
    print("The number of unknown images are "
          + str(unknown) + " out of " + str(file_count)
          + " ---> " + str(unknown/file_count * 100) + " %")

    return unknown/file_count

def execute(img, saved_model):
    model = load_model(saved_model)
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    class_names = ['FOOD', 'NONFOOD']

    predictions = model.predict(img_array)
    conf_int = [0.4, 0.6]
    if predictions[0] < conf_int[0]:
        print('Image is FOOD')
        result = 'FOOD'
    elif predictions[0] > conf_int[1]:
        print('Image is NONFOOD')
        result = 'NONFOOD'
    else:
        print('This image most likely belongs to {} with a {:.2f} percent confidence.'
              .format(class_names[0], 100 - 100 * np.max(predictions[0])))
        result = 'NONE'
    print(class_names[0], 100 - 100 * np.max(predictions[0]))

    return result
