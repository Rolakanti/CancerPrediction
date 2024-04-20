import os
import numpy as np
import numpy as np
import pickle
import cv2
from keras.applications.vgg19 import VGG19
import keras
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array
from keras.metrics import Recall,Precision
from tensorflow.keras.utils import img_to_array,load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from DBConfig import DBConnection
from keras.preprocessing import image
import sys
def build_vgg19():
    try:
        database = DBConnection.getConnection()
        cursor = database.cursor()
        EPOCHS = 5
        BS = 32

        print("[INFO] Loading Training dataset images...")
        DIRECTORY = "..\\Prediction\\dataset"
        CATEGORIES=['Adenocarcinoma','Benign','Squamous_Carcinoma']


        image_data = []
        target_class = []

        for category in CATEGORIES:
            print(category)
            path = os.path.join(DIRECTORY, category)
            print(path)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                img = load_img(img_path, target_size=(128,128))
                img = img_to_array(img)
                #img = img / 255
                image_data.append(img)
                target_class.append(category)

        label_binarizer = LabelBinarizer()
        image_labels = label_binarizer.fit_transform(target_class)
        pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
        n_classes = len(label_binarizer.classes_)
        print(n_classes)
        np_image_list = np.array(image_data, dtype=np.float16) / 225.0

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

        # Model Initialization
        base_model=VGG19(include_top=False,input_shape=(128,128,3))
        base_model.trainable=False

        classifier=keras.models.Sequential()
        classifier.add(base_model)
        classifier.add(Flatten())
        classifier.add(Dense(3,activation='softmax'))

        classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',Recall(),Precision()])

        print("[INFO] training network...")

        aug = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, 
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")
        

        history = classifier.fit_generator(
            aug.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
        )

        acc = history.history['accuracy']
        '''val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        precision=history.history['precision']
        val_precision = history.history['val_precision']

        recall = history.history['recall']
        val_recall = history.history['val_recall']'''

        epochs = range(1, len(acc) + 1)
        # Train and validation accuracy
        plt.plot(epochs, history.history['accuracy'], 'b', label='Training accurarcy')
        plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation accurarcy')
        plt.title('VGG19 Model Accurarcy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.savefig('static/accuracy.png')
        plt.show()



        # Train and validation loss
        plt2.plot(epochs,  history.history['loss'], 'b', label='Training loss')
        plt2.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
        plt2.title('VGG19 Model Loss')
        plt2.ylabel('loss')
        plt2.xlabel('epoch')
        plt2.legend()
        plt2.savefig('static/loss.png')
        plt2.show()

        plt3.plot(epochs, history.history['precision'],'b', label='Training loss')
        plt3.plot(epochs, history.history['val_precision'], 'r', label='Validation loss')
        plt3.title('VGG19 Model Precision')
        plt3.ylabel('precision')
        plt3.xlabel('epoch')
        plt3.legend(['Training', 'Validation'], loc='lower right')
        plt3.savefig('static/precision.png')
        plt3.show()

        plt4.plot(epochs,history.history['recall'],'b')
        plt4.plot(epochs,history.history['val_recall'],'r')
        plt4.title('VGG19 Model Recall')
        plt4.ylabel('recall')
        plt4.xlabel('epoch')
        plt4.legend(['Training', 'Validation'], loc='lower right')
        plt4.savefig('static/recall.png')
        plt4.show()


        print("[INFO] Calculating VGG19 model accuracy")
        scores = classifier.evaluate(x_test, y_test)
        print("Test Accuracy:",scores)
        #vgg19_accuracy=scores[1]*100
        #print(vgg19_accuracy)'''
        print("Training Completed..!")

        loss = scores[0]

        acc = scores[1]

        recall =scores[2]

        precision=scores[3]


        
        
        # save the model to disk
        #print("[INFO] Saving model...")
        classifier.save('vgg19_model.h5')
        cursor.execute("delete from evaluations")

        sql = "insert into evaluations values('"+str(acc)+"','"+str(loss)+"','"+str(precision)+"','"+str(recall)+"')"
        cursor.execute(sql)
        database.commit()
       

        
    except Exception as e:
        print("Error=" , e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

build_vgg19()
