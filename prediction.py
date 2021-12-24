
from tensorflow.keras.utils import get_file, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow import expand_dims, nn
import numpy as np
import os

def to_predict(dir_new, saved_model, h, w,):

    image_size = (h, w) # image size (resize) can affect the accuracy !
    class_names = ['FOOD', 'NONFOOD']
    nonfood_cnt = 0
    food_cnt = 0
    conf_int = [0.4, 0.6]
    unknown = 0

    path, dirs, files = next(os.walk(dir_new))
    file_count = len(files)
    # Load the model
    model = load_model(saved_model)

    for image_nr in range(1, file_count+1):
        # classify new pictures (not given neither during training and validation) using the model
        img_name = 'image ' + '(' + str(image_nr) + ').jpg'
        img = load_img(dir_new + '/' + img_name, target_size=image_size)
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        if predictions[0] < conf_int[1]:
            nonfood_cnt = nonfood_cnt + 1
        elif predictions[0] > conf_int[0]:
            food_cnt = food_cnt + 1
        else:
            unknown = unknown + 1
        print(img_name, "most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_names[0], 100 - 100 * np.max(predictions[0])))

    print("The number of images associated to FOOD are "
          + str(food_cnt) + " out of " + str(file_count)
          + " ---> " + str(food_cnt/file_count * 100) + " %")
    print("The number of images associated to NON_FOOD are "
          + str(nonfood_cnt) + " out of " + str(file_count)
          + " ---> " + str(nonfood_cnt/file_count * 100) + " %")
    print("The number of unknown images are "
          + str(unknown) + " out of " + str(file_count)
          + " ---> " + str(unknown/file_count * 100) + " %")
