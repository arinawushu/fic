"""
Project:    The idea is to implement a SVM which is able to identify and separate different classes:
            for instance FOOD and NON-FOOD.
"""
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from sklearn.metrics import confusion_matrix
import os
from natsort import os_sorted



def svm_classifier(directory, dir_proj, h, w, d):

    categories = ['FOOD', 'NONFOOD']

    """
        Loading the images:
        - Please download all the images and put them into a directory with two sub-directory: FOOD and NONFOOD
        - The user may select the right directory on his laptop
    """

    flat_data = []  # initialize an array for flattening data
    categor_arr = []  # initialize an array for categories
    t0 = time()

    for i in categories:
        print(f'loading a category: {i}')
        path = os.path.join(directory, i)  # loading each folder as category
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (h, w, d))  # resizing image
            flat_data.append(img_resized.flatten())  # flattening data of images
            categor_arr.append(categories.index(i))  # putting categories in array
        print(f'Category {i} loaded')
    print('All categories are loaded')

    flat_data = np.array(flat_data)
    categor_arr = np.array(categor_arr)
    data_framed = pd.DataFrame(flat_data)
    data_framed['By category'] = categor_arr

    '''this will be the whole data we work with: number of rows is number of images, columns - size
        and categories from 0 to 1 since we have only 2 of them
        
        it's possible to increase amount of categories and classify different types of food'''

    print(data_framed)

    '''Now we have to split data to training and testing. Let's split 80% for training and 20% for testing'''
    x = data_framed.iloc[:, :-1]  #part with images
    y = data_framed.iloc[:, -1]  #only categories
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=True)
    print('Splitting is done')

    '''Training data using sklearn C-Support Vector Classification with cross-validation
    params = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}  # parameter grid for cross validation
    svc = GridSearchCV(SVC(kernel='rbf', verbose=True, class_weight="balanced", probability=True), params)
    model = svc.fit(x_train, y_train)
    print('Best estimator found by grid search: ', svc.best_estimator_, 'time: %0.3fs' % (time() - t0))'''

    '''Best estimator found by grid search:  SVC(C=1000.0, class_weight='balanced', gamma=0.001, probability=True,
    verbose=True)'''

    svc = svm.SVC(C=1000, gamma=0.001, kernel='rbf', verbose=True, class_weight="balanced", probability=True)
    model = svc.fit(x_train, y_train)

    '''Prediction on test images'''
    y_pred = model.predict(x_test)

    '''Quantitative evaluation of the model quality on the test set'''
    accuracy = accuracy_score(y_pred, y_test) * 100
    print(f'The model accuracy is {accuracy}%')

    '''Confusion matrix and classification report'''
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(2)))

    '''Saving the model'''
    pickle.dump(model, open('img_model.p', 'wb'))
    print('Model is saved')

    return accuracy, (time() - t0)


def to_predict(dir_new, saved_model, h, w,):
    class_names = ['FOOD', 'NONFOOD']
    flat_img = []  # initialize an array for flattening data
    model = pickle.load(open(saved_model, 'rb'))
    conf_int = [0.4, 0.6]
    nonfood_cnt = 0
    food_cnt = 0
    unknown = 0
    # a_0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #        1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    #        1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    #        1, 1, 1]

    print('Reading images')

    sort = os_sorted(os.listdir(dir_new))
    print(sort)

    for img in sort:
        img_array = plt.imread(os.path.join(dir_new, img))
        img_resized = resize(img_array, (h, w, 3))  # resizing image
        flat_img.append(img_resized.flatten())  # flattening data of images

    predictions = model.predict_proba(flat_img)
    a = predictions[:, 0]
    img_count = len(a)

    for i in range(0, img_count):
        if a[i] > conf_int[1]:
            food_cnt = food_cnt + 1
        elif a[i] < conf_int[0]:
            nonfood_cnt = nonfood_cnt + 1
        else:
            unknown = unknown + 1

    #Building confusion matrix
    #print(confusion_matrix(a_0, model.predict(flat_img)))

    print("The number of images associated to FOOD are "
          + str(food_cnt) + " out of " + str(img_count)
          + " ---> " + str(food_cnt / img_count * 100) + " %")
    print("The number of images associated to NON_FOOD are "
          + str(nonfood_cnt) + " out of " + str(img_count)
          + " ---> " + str(nonfood_cnt / img_count * 100) + " %")
    print("The number of unknown images are "
          + str(unknown) + " out of " + str(img_count)
          + " ---> " + str(unknown / img_count * 100) + " %")

    plt.hist(100-a*100, 10, facecolor='blue', alpha=0.5)
    plt.title("Histogram:" + class_names[1])  # change the class name for the histogram
    plt.xlabel('confidence')
    plt.ylabel('samples')
    plt.ylim(0, 100)
    plt.xlim(0, 100)

    plt.show()

    return unknown/img_count


def execute(img, saved_model):
    categories = ['FOOD', 'NONFOOD']
    model = pickle.load(open(saved_model, 'rb'))
    im = [img.flatten()]

    probability = model.predict_proba(im)
    for ind, val in enumerate(categories):
        print(f'{val} = {probability[0][ind] * 100}%')

    result = categories[model.predict(im)[0]]
    print("The predicted image is : " + categories[model.predict(im)[0]])

    return result
