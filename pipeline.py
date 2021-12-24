'''
This is a file with the main pipeline for the project
'''

'''Import of python libraries'''
from skimage.transform import resize
from skimage.io import imread
from random import seed
from random import randint
from PIL import Image
import matplotlib.pyplot as plt

'''Import of our functions'''
import food_classifier as cnn_cl
import svm_classifyer as svm_cl


def hello(user):

    if user == 'Cairo':
        data_directory = 'C:/Users/cog1/Downloads/all' # directory of training and testing sets Cairo
    elif user == 'Arina':
        data_directory = '/Users/ArinaShvetsova/food_image_classifier/training' # directory of training and testing sets Arina
    elif user == 'Marco':
        data_directory = input('Marco, please past your directory with images: ')
    else:
        data_directory = input(f'{user}, please input your directory: ')

    return data_directory

def classify(classifier, data_directory, project_directory, h, w, d):

    if classifier == '1':
        accuracy, time_elapsed = cnn_cl.cnn_classifier(data_directory, project_directory, h, w, d)
        print('Accuracy: ', accuracy, 'Time elapsed: ', time_elapsed)
    elif classifier == '2':
        accuracy, time_elapsed = svm_cl.svm_classifier(data_directory, project_directory, h, w, d)
        print('Accuracy: ', accuracy, 'Time elapsed: ', time_elapsed)
    else:
        print('No classifier was chosen')

    return 1

def predict(user, classifier, project_directory, h, w):

    if user == 'Cairo':
        dir_new = 'C:/Users/cog1/Downloads/to_predict/'  # new images directory Cairo
        if classifier == '1':
            saved_model = 'model_cnn_final.h5'
            cnn_cl.to_predict(dir_new, saved_model, h, w)
        elif classifier == '2':
            saved_model = 'img_model.p'
            svm_cl.to_predict(dir_new, saved_model, h, w)
    elif user == 'Arina':
        dir_new = project_directory + '/prediction_by_label/NONFOOD/'  # new images directory Arina
        if classifier == '1':
            saved_model = 'model_cnn_final.h5'
            cnn_cl.to_predict(dir_new, saved_model, h, w)
        elif classifier == '2':
            saved_model = 'img_model.p'
            svm_cl.to_predict(dir_new, saved_model, h, w)
    elif user == 'Marco':
        dir_new = input('Marco, please past your directory with new images')
        if classifier == '1':
            saved_model = 'model_cnn_final.h5'
            cnn_cl.to_predict(dir_new, saved_model, h, w)
        elif classifier == '2':
            saved_model = 'img_model.p'
            svm_cl.to_predict(dir_new, saved_model, h, w)
    else:
        dir_new = input('Please input your directory with new images')
        if classifier == '1':
            saved_model = 'model_cnn_final.h5'
            cnn_cl.to_predict(dir_new, saved_model, h, w)
        elif classifier == '2':
            saved_model = 'img_model.p'
            svm_cl.to_predict(dir_new, saved_model, h, w)

    return 1


def is_that_food(url, classifier, data_directory, h, w):
    img = imread(url)
    img_resize = resize(img, (h, w, 3))

    if classifier == '1':
        saved_model = 'model_cnn_final.h5'
        res = cnn_cl.execute(img_resize, saved_model)
    elif classifier == '2':
        saved_model = 'img_model.p'
        res = svm_cl.execute(img_resize, saved_model)

    plt.imshow(img)
    plt.title(f'This image is {res}')
    # plt.title(f'This image is {res} with classifier {classifier}')
    plt.show()

    seed(42)
    value = randint(0, 1000)
    pred_im = input('Is the prediction right? y/n ')
    if pred_im == 'y':
        print('Congratulations')
    else:
        save = input('What is this image? food-0, nonfood-1 ')
        if save == '1':
            print('Saving image in food directory')
            image_path = data_directory + '/FOOD'
            img = Image.fromarray(img, 'RGB')
            img.save(f"{image_path}/image{value}.png")
        else:
            print('Saving image in non-food directory')
            image_path = data_directory + '/NONFOOD'
            img = Image.fromarray(img, 'RGB')
            img.save(f"{image_path}/image{value}.png")

    return 1

