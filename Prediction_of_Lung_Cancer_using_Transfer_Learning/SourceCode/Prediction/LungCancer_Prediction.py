

import os
import sys
import numpy as np
import operator
import pickle
from keras.models import Sequential, load_model
import cv2
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing import image
from tensorflow.keras.utils import load_img

def prediction_image(test_image):
    try:

        data = []
        img_path = test_image

        testing_img=cv2.imread(img_path)
        cv2.imwrite("../Prediction/static/lr_detection.jpg", testing_img)

        model_path = 'vgg19_model.h5'
        model = load_model(model_path)

        test_image = load_img(test_image, target_size=(128, 128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        prediction = model.predict(test_image)
        lb = pickle.load(open('label_transform.pkl', 'rb'))
        prediction_result=lb.inverse_transform(prediction)[0]

        print(prediction_result)

        return prediction_result





    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)

'''testimage="lungaca27.jpeg"
prediction_image(testimage)'''

