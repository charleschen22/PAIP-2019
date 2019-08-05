# coding: utf-8

# In[ ]:


from scipy.misc import imsave
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from skimage import io
from keras.utils.np_utils import to_categorical
from pydaily import filesystem
from datetime import datetime
import staintools
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from preprocessing import get_id_list
from patches_production import gen_imgs, get_start_coordinate, crop
# set the device to run the model
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
KTF.set_session(sess)

# get id list
id_list = get_id_list('./data/OriginalImage')

print('Patches producing can take a while...')
svs_dir_ = './data/OriginalImage'
viable_dir_ = './data/ViableMask'
whole_dir_ = './data/WholeMask'

svs_paths = glob.glob(osp.join(svs_dir_, '*.svs'))
svs_paths.extend(glob.glob(osp.join(svs_dir_, '*.SVS')))
viable_paths = glob.glob(osp.join(viable_dir_, '*.tif'))
whole_paths = glob.glob(osp.join(whole_dir_, '*tif'))

if os.path.exists('./data/coordinates.csv'):
    sample_total = pd.read_csv('./data/coordinates.csv', index_col=0)
else:
    sample_total = pd.DataFrame([])
    i = 0
    while i < len(id_list):
        print('Processing data #' + str(i) + '............')
        patches = get_start_coordinate(id_list[i], (512, 512))
        patches['id'] = id_list[i]
        sample_total = sample_total.append(patches, ignore_index=True)
        i += 1
    sample_total.to_csv('./data/coordinates.csv')
    sample_total = pd.read_csv('./data/coordinates.csv', index_col=0)

# devide whole samples into 80% training set and 20% validation set
sample_total = sample_total.rename(columns={'0': 'x', '1': 'y'})
sample_total[['x']] = sample_total[['x']].astype(str)
sample_total[['y']] = sample_total[['y']].astype(str)
train_index = np.random.randint(0, sample_total.shape[0], int(0.8*sample_total.shape[0]))
valid_index = np.delete(np.array(range(sample_total.shape[0])), train_index)
train_samples = sample_total.loc[train_index]
valid_samples = sample_total.loc[valid_index]


# build the model
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Lambda(lambda x: x / 511.0 - 0.5, input_shape=(512, 512, 4)))
model.add(Convolution2D(100, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(200, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
model.add(Convolution2D(300, (3, 3), activation='elu',  padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(2, (1, 1))) # this is called upscore layer for some reason?
model.add(Conv2DTranspose(2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 32
N_EPOCHS = 10
crop_size = (512, 512)
# checkpoint
filepath = "./data/model/fcn_weight_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print("Training starts...")

model.fit_generator(gen_imgs(train_samples, crop_size, BATCH_SIZE, 'viable', color_norm=False), np.ceil(len(train_index) / BATCH_SIZE), epochs=N_EPOCHS)

# save the model
model.save('./model/modelfcn.h5')
model_json = model.to_json()
with open("./model/modelfcn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("modelunet.h5")
print("Saving model done")

