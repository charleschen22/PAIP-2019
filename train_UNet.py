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
import tensorflow as tf
import staintools
import keras.backend.tensorflow_backend as KTF
from preprocessing import get_id_list
from patches_production import crop, gen_imgs, gen_imgs_random
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
KTF.set_session(sess)

id_list = get_id_list('./data/OriginalImage')

print('Patches producing can take a while...')
svs_dir_ = './data/OriginalImage'
viable_dir_ = './data/ViableMask'
whole_dir_ = './data/WholeMask'

#slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
svs_paths = glob.glob(osp.join(svs_dir_, '*.svs'))
svs_paths.extend(glob.glob(osp.join(svs_dir_, '*.SVS')))
viable_paths = glob.glob(osp.join(viable_dir_, '*.tif'))
whole_paths = glob.glob(osp.join(whole_dir_, '*tif'))


from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights = None,input_size = (512, 512, 4)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


model = unet()

BATCH_SIZE = 16 #oom when batch size equal to 32
N_EPOCHS = 3
crop_size = (512, 512)
# checkpoint
filepath = "./data/model/random_unet_weight_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print("Training starts...")

id_list_train = random.sample(id_list, 40)
for i in id_list_train:
    id_list.remove(i)
id_list_test = id_list
print(id_list_test)
model.fit_generator(gen_imgs_random(id_list_train, crop_size, BATCH_SIZE, 'viable', color_norm=False), np.ceil(8000 / BATCH_SIZE),
    epochs=N_EPOCHS)

# save the model
model.save('./model/random_model_unet.h5')
model_json = model.to_json()
with open("./model/random_model_unet.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("modelunet.h5")
print("Saving model done")
