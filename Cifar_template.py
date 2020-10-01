from keras import layers 
from keras import models 
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from keras import backend as k
import tensorflow as tf
 
import os 
import matplotlib.pyplot as plt 
 
# last edited 19/08/2020
# Instantiating a small CNN  
def MakeModel(weights_path=None):
	model = models.Sequential() 
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) 
	model.add(layers.MaxPooling2D((2, 2))) 
	#-----------------------------------
	#-------Build your layers here -----
	#-----------------------------------
	model.add(layers.Flatten()) 
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(512, activation='relu')) 
	model.add(layers.Dense(4, activation='softmax', name='output')) 
	#model.add(layers.Dense(1000, activation='softmax'))
 
	return model
	
	
base_dir = 'E:\Deeplearning\Cifar\Cifar4_SortedImages' 
train_dir = os.path.join(base_dir, 'train') 
validation_dir = os.path.join(base_dir, 'validation') 
test_dir = os.path.join(base_dir, 'test') 

model = MakeModel()

# Displaying a summary of the model 
model.summary() 

# Configuring our model for training 
# For a binary classification problem 
model.compile(loss='binary_crossentropy', 
    optimizer=optimizers.RMSprop(lr=1e-4), 
    metrics=['acc']) 
	
	# Using ImageDataGenerator to read images from directories 
# all images will be rescaled by 1./255 
#train_datagen = ImageDataGenerator(rescale=1./255)  
test_datagen = ImageDataGenerator(rescale=1./255) 
validation_datagen = ImageDataGenerator(rescale=1./255) 

# Setting up a data augmentation configuration via
# ImageDataGenerator
train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=0,
 width_shift_range=0.0,
 height_shift_range=0.0,
 shear_range=0.0,
 zoom_range=0.0,
 horizontal_flip=False,
 fill_mode='nearest')
 
train_generator = train_datagen.flow_from_directory( 
    train_dir, # this is the target directory 
    target_size=(224, 224),  
    batch_size=1) 
    # since we use binary_crossentropy loss, we need binary labels 
 
validation_generator = validation_datagen.flow_from_directory( 
    validation_dir, 
    target_size=(224, 224), 
    batch_size=1)
	
	# Fitting our model using a batch generator 
# Trains the model for a fixed number of epochs (iterations on a dataset). 
history = model.fit_generator( 
   train_generator, 
   steps_per_epoch=1, 
   epochs=10,  
   validation_data=validation_generator, 
   validation_steps=1) 
 
 
 # Evaluating the model - test on finalised model 
# Returns the loss value & metrics values for the model in test mode. 
test_generator = test_datagen.flow_from_directory( 
    test_dir, 
    target_size=(224, 224), 
    batch_size=20) 
    #class_mode='binary') 
 
# finally evaluate this model on the test data 
results = model.evaluate_generator( 
    test_generator, 
    steps=1000) 
	
print('Final test accuracy:', (results[1]*100.0))
	
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
 