from tensorflow.keras.utils import img_to_array,load_img
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing import image as image_utils
import numpy
import pickle
from keras.preprocessing import image
import numpy as np

model_path = 'vgg19_model.h5'
model = load_model(model_path)
testimage="lungaca27.jpeg"
test_image = load_img(testimage, target_size=(128, 128))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255
prediction = model.predict(test_image)
lb = pickle.load(open('label_transform.pkl', 'rb'))
print(lb.inverse_transform(prediction)[0])