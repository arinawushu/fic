import os
import sys
import pipeline as pipl

'''
    Project:    Main file to launch for both classifiers SVM and CNN:

    - First part is the training with one of the classifiers and prediction
    - Second part is testing an algorithm with set of images from friends or random image from internet
    - Third part is reteaching algorithm with new image

    Authors:    Giuliano Cairoli - giuliano.cairoli@students.unibe.ch
                Marco Wyss - marco.wyss1@students.unibe.ch
                Arina Shvetsova - arina.shvetsova@students.unibe.ch
'''


def main():
    sys.stderr = open("errors.txt", "w")
    user = input('Welcome to food/non-food classifier! Please indicate the user: Cairo, Marco, Arina,'
                 ' or your name: ')
    print(f'Hello {user}!')
    project_directory = os.getcwd()
    # data_directory = pipl.hello(user)
    data_directory = project_directory + '/training'

    choice_of_image_size = input('Image size will be by default 64x48, do you want to change? y/n ')
    choice_of_image_size = 'n'
    '''Assigning global variables - size of images'''
    if choice_of_image_size == 'y':
        h, w = input('Input image size: h w')
        d = 3  # type of image
    else:
        h = 64 # height
        w = 48 # width
        d = 3  # type of image

    '''Classification with 2 classifiers'''
    classifier = input('Choose your warrior 1 - cnn classifier, 2 - svm classifier ')
    sys.stdout = open("output.txt", "w")
    print(f'User is {user}, Image size is {h}x{w}, Classifier is {classifier}')
    pipl.classify(classifier, data_directory, project_directory, h, w, d)

    '''Prediction with new set of images'''
    pipl.predict(user, classifier, project_directory, h, w)
    sys.stdout.close()

    '''Verifying image from internet'''
    url = input('Enter URL of Image: ')
    pipl.is_that_food(url, classifier, data_directory, h, w)

    '''Retraining the model'''
    answer = input('Retrain the model? y/n ')
    if answer == 'y':
        print('Retraining the model')
        pipl.classify(classifier, data_directory, project_directory, h, w, d)
    else:
        print('Thank you, bye ~')

    sys.stderr.close()

if __name__ == "__main__":
    """The program's entry point."""

    main()
