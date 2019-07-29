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


def rgba_2_rgb(r, g, b, a, rb=255, gb=255, bb=255):
    r3 = int((1-a)*rb + a*r)
    g3 = int((1-a)*gb + a*g)
    b3 = int((1-a)*bb + a*b)

    return(np.array([r3, g3, b3]))


def get_id_list(slides_dir):
    id_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    id_list.extend([os.path.basename(ele) for ele in svs_file_list])
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    id_list.extend([os.path.basename(ele) for ele in SVS_file_list])
    ids = [os.path.splitext(ele)[0] for ele in id_list]

    return ids


id_list = get_id_list('./data/OriginalImage')


def sequential_crop(img_id, start, crop_size):

    start_x, start_y = start
    slide_path = './data/OriginalImage/' + str(img_id) + '.svs'
    if not os.path.exists(slide_path):
        slide_path = './data/OriginalImage/' + str(img_id) + '.SVS'
    slide = openslide.open_slide(slide_path)
    croped_slide_img = slide.read_region((start_x, start_y), 0, crop_size)
    croped_slide_img = np.array(croped_slide_img)#[:, :, 0:3]
    mask_path = './data/ViableMask/' + str(img_id) + '_viable.tif'
    mask = io.imread(mask_path)
    croped_mask_img = mask[start_x:start_x+crop_size[0], start_y:start_y+crop_size[1]]

    return(croped_slide_img, croped_mask_img, start)


def get_start_coordinate(img_id, crop_size):
    coordinate = pd.DataFrame([])
    mask_path = './data/ViableMask/' + str(img_id) + '_viable.tif'
    mask = io.imread(mask_path)
    mask_shape = np.shape(mask)
    img_num_wide = mask_shape[0]//crop_size[0]
    img_num_height = mask_shape[1]//crop_size[1]
    for i in range(img_num_height):
        for j in range(img_num_wide):
            x = j*crop_size[0]
            y = i*crop_size[1]
            coordinate = coordinate.append([[str(x), str(y)]])
    coordinate = coordinate.append([[str(mask_shape[0]-crop_size[1]+1), str(mask_shape[1]-crop_size[0]+1)]])
    print(coordinate)

    return(coordinate)


print('Patches producing can take a while...')
svs_dir_ = './data/OriginalImage'
viable_dir_ = './data/ViableMask'
whole_dir_ = './data/WholeMask'

#slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
svs_paths = glob.glob(osp.join(svs_dir_, '*.svs'))
svs_paths.extend(glob.glob(osp.join(svs_dir_, '*.SVS')))
viable_paths = glob.glob(osp.join(viable_dir_, '*.tif'))
whole_paths = glob.glob(osp.join(whole_dir_, '*tif'))

'''
sample_total = pd.DataFrame([])
i = 0
while i < len(id_list):
    print('Processing data #' + str(i) + '............')
    patches = get_start_coordinate(id_list[i], (512, 512))
    patches['id'] = id_list[i]
    sample_total = sample_total.append(patches, ignore_index=True)

    i += 1

sample_total.to_csv('./data/coordinates.csv')
'''

def gen_imgs(samples, batch_size, shuffle=True, color_norm=True, target="./data/svs_patches/01_01_0091_12800_22528.png"):

    save_svs_patches = './data/svs_patches'
    save_viable_patches = './data/viable_patches'
    num_samples = len(samples)
    target = staintools.read_image(target)

    while 1:
        if shuffle:
            samples = samples.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]
            images = []
            masks = []
            id_list = list(batch_samples['id'])
            x_list = list(batch_samples['x'])
            y_list = list(batch_samples['y'])

            for i in range(batch_size):
                a = id_list[i]
                print('********** Crop in ' + str(a) + ' **********')
                cor1 = int(x_list[i])
                cor2 = int(y_list[i])
                slide_patch, mask_patch = sequential_crop(a, (cor1, cor2), (512, 512))[0:2]

                if color_norm:
                    target = staintools.LuminosityStandardizer.standardize(target)
                    slide_patch = staintools.LuminosityStandardizer.standardize(slide_patch)
                    normalizer = staintools.StainNormalizer(method='vahadane')
                    normalizer.fit(target)
                    slide_patch = normalizer.transform(slide_patch)

                if not os.path.exists(save_svs_patches):
                    os.mkdir(save_svs_patches)
                imsave(osp.join(save_svs_patches, str(a) + '_' + str(cor1) + '_' + str(cor2) + '.png'), slide_patch)
                if not os.path.exists(save_viable_patches):
                    os.mkdir(save_viable_patches)
                imsave(osp.join(save_viable_patches, str(a) + '_' + str(cor1) + '_' + str(cor2) + '_viable.png'), mask_patch)

                #mask_patch = np.resize(mask_patch, [512, 512, 1])
                images.append(slide_patch)
                masks.append(mask_patch)

                batch_samples = pd.DataFrame(batch_samples)

                X_train = np.array(images)
                y_train = np.array(masks)
                y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], 512, 512, 2)
                print(np.shape(X_train))
                print(np.shape(y_train))

            yield X_train, y_train


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


sample_total = pd.read_csv('./data/coordinates.csv', index_col=0)
sample_total = sample_total.rename(columns={'0': 'x', '1': 'y'})
sample_total[['x']] = sample_total[['x']].astype(str)
sample_total[['y']] = sample_total[['y']].astype(str)
train_index = np.random.randint(0, sample_total.shape[0], int(0.8*sample_total.shape[0]))
valid_index = np.delete(np.array(range(sample_total.shape[0])), train_index)
train_samples = sample_total.loc[train_index]
valid_samples = sample_total.loc[valid_index]


BATCH_SIZE = 32
N_EPOCHS = 10

# checkpoint
filepath="./data/model/fcn_weight_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print("Training starts...")

model.fit_generator(gen_imgs(train_samples, BATCH_SIZE, color_norm=False), np.ceil(len(train_index) / BATCH_SIZE),
    validation_data=gen_imgs(valid_index, BATCH_SIZE),
    validation_steps=np.ceil(len(valid_index) / BATCH_SIZE),
    epochs=N_EPOCHS)

model.save('./model/modelfcn.h5')
model_json = model.to_json()
with open("./model/modelfcn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("modelunet.h5")
print("Saving model done")