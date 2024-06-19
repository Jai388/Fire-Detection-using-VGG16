import os  
import cv2                                   
import numpy as np                            
from PIL import Image                # Python image library 
import matplotlib.pyplot as plt      # For data Visualization-plots
 
from IPython.display import display  # For display
import numpy as np
import matplotlib.pyplot as plt
import warnings                      # To ignore unnecessary python warnings 
warnings.filterwarnings('ignore')
     

# Here we are using VGG16 model
from keras.applications import VGG16                                            
from keras.callbacks import ModelCheckpoint                # further to train the saved model 
from keras import models, layers, optimizers               # building DNN
from tensorflow.keras.models import load_model             # To load saved model 
from keras.preprocessing.image import ImageDataGenerator     
from imutils import paths 
     

#classes = 2
conv_layer = VGG16(weights='imagenet',             # Base layer
                 include_top=False,
                 input_shape=(224, 224, 3))

model = models.Sequential()
model.add(conv_layer)
model.add(layers.Flatten())
#model.add(layers.Dense(classes))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_layer.trainable = False

model.compile(loss='binary_crossentropy', 
             optimizer=optimizers.RMSprop(lr=1e-4), 
             metrics=['accuracy'])
     
model.summary()

train_path = r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\train"
train_fire = r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\train\train_fire"
train_nofire =r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\train\train_nofire"
test_path = r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\test"
test_fire =r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\test\test_fire"
test_nofire =r"C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\firedetection\test\test_nofire"
Train_datagen = ImageDataGenerator(rescale=1./255, 
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

Test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = Train_datagen.flow_from_directory(train_path, 
                                                   target_size=(224,224), 
                                                   batch_size=32,
                                                   class_mode='binary')

test_generator = Test_datagen.flow_from_directory(test_path, 
                                                   target_size=(224,224), 
                                                   batch_size=32,
                                                   class_mode='binary')
H = model.fit(train_generator, epochs=30, 
                     validation_data=test_generator)


# Assuming H is the history object returned by model.fit()

N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
model.save('Optimized_model.h5') 

