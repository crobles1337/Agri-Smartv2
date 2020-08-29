#takes comopressed npz file, loads appropriates, inputs for training, de-loads back to compression, picks up next batch of files
#from tensorflow import keras
#from keras.models import Sequential
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import rasterio
from rasterio import plot
import numpy as np
from numpy import load

#NDVILOAD = load('NDVICOMPRESSEDTEST.npz')
#NDVIARRAY = NDVILOAD['arr_0']
#plot.show(NDVIARRAY, cmap='RdYlGn')



#datagen = ImageDataGenerator()

#train_it = datagen.flow_from_directory('NPZs/data/train', class_mode='binary', batch_size = 30)



loads = load('Bundle2Test\SoilStats.npz')
loads2 = load('Bundle2Test\SoilUncertainty.npz')
loadload = loads['arr_0']
load2load = loads2['arr_0']
print(loadload)
print(load2load)