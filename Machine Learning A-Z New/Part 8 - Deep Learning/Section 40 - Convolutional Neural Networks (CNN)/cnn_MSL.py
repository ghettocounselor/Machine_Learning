# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Lecture 252 https://www.udemy.com/machinelearning/learn/lecture/6171688
# There are 10 lectures

# Part 1 - Building the CNN

# =============================================================================
# # Importing the Keras libraries and packages
# =============================================================================
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# =============================================================================
# # Initialising the CNN
# =============================================================================
# create an object of the Sequential classStep
classifier = Sequential()

# =============================================================================
# # Step 1 - Convolution
# =============================================================================
# Lecture 255 https://www.udemy.com/machinelearning/learn/lecture/6210016
# The steps to work through to tune the algorithm. 
from PIL import Image
img = Image.open("CNN_Step1_Convolution.png")
img.show()
# we'll add a layer using Conv2D; parameters 
# 32 is the filters (feature detectors)
# 3 and 3 are the sizes of the feature detectors 3 x 3
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

from PIL import Image
# remembering what the images are converted too; above 64x64 is size of image and 3 is
# the number of channels. 
img = Image.open("InputImagesConversion.png")
img.show()

# =============================================================================
# # Step 2 - Pooling
# =============================================================================
# Lecture 256 https://www.udemy.com/machinelearning/learn/lecture/6213362
# we pool to reduce the number of layers in our collection of feature layers
from PIL import Image
img = Image.open("CNN_Step2_MaxPooling.png")
img.show()
from PIL import Image
img = Image.open("CNN_Step2_PoolingLayer.png")
img.show()
# the pool_size is the size of the 'tool' we slide over the feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer - to improve the accuracy with test site
# Lecture 261 https://www.udemy.com/machinelearning/learn/lecture/6224514
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# =============================================================================
# # Step 3 - Flattening
# =============================================================================
# Lecture 257 https://www.udemy.com/machinelearning/learn/lecture/6213932
from PIL import Image
img = Image.open("CNN_Step3_Flattening.png")
img.show()
from PIL import Image
img = Image.open("CNN_Step3_Flattening2.png")
img.show()
classifier.add(Flatten())

# =============================================================================
# # Step 4 - Full connection
# =============================================================================
from PIL import Image
img = Image.open("CNN_Step4_FullConnection.png")
img.show()
# The Dense function is what we use to create a fully connected layer
# units: Positive integer, dimensionality of the output space. The # of nodes in the output layer
classifier.add(Dense(units = 128, activation = 'relu')) # gives probabilities
classifier.add(Dense(units = 1, activation = 'sigmoid')) # yes or no 

# =============================================================================
# Full picture of our CNN
# =============================================================================
from PIL import Image
img = Image.open("CNN_4Steps.png")
img.show()

# =============================================================================
# # Compiling the CNN
# =============================================================================
# The optimizer is the stochastic decent algorithm, we'll use one called 'adam'
# loss is the loss function within the 'adam' algorithm. This loss function will be 
# optimized by the NN, we have a binary outcome so we'll use binary. 
# Metrics is the criteria that will be used to improve the performance. 
# we're telling the NN to use the accuracy as the criteria for improvement. 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# =============================================================================
# # Part 2 - Fitting the CNN to the images - 'image augmentation'
# =============================================================================
# First we need to do 'image augmentation' to protect against overfitting
# https://keras.io/preprocessing/image/ 
# we used the Flow method
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/

training_set = train_datagen.flow_from_directory('/Users/markloessi/Machine_Learning_hugeDataset/dataset/training_set',
                                                 target_size = (64, 64), # we choose 64 x 64 above
                                                 batch_size = 32,
                                                 class_mode = 'binary') # again we are binary

test_set = test_datagen.flow_from_directory('/Users/markloessi/Machine_Learning_hugeDataset/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# =============================================================================
# Fitting the CNN to our augmented set of images
# =============================================================================
# also testing performance
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # num of images in training set
                         epochs = 1, # the number we choose - in class they used 25 :~/ but poor little computer
                    # it took 3.5 hours to finish 1 epoch
                         validation_data = test_set, # where to evaluate the performance of our CNN
                         validation_steps = 2000) # num of images in our test set

'''
Epoch 1/1
8000/8000 [==============================] - 10717s 1s/step - loss: 0.3762 - acc: 0.8223 - val_loss: 0.5815 - val_acc: 0.8006
Out[1]: <keras.callbacks.History at 0xb2f5c7e48>
'''






