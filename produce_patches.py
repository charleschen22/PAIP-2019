
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
from PIL import Image
import cv2
from keras.utils.np_utils import to_categorical
from skimage import io
import staintools
import cv2


def gen_imgs(samples, batch_size, shuffle=True, color_norm=False, target="./data/svs_patches/01_01_0091_12800_22528.png"):

    '''
    :param samples: a dataframe which contains the top left coordinates of all patches which contain at least 50% tissue from all images
    :param batch_size: an int stands for size of the batch
    :param shuffle: an option whether shuffle samples
    :param color_norm: an options whether do color normalization
    :param target: the path of the base image to do color normalization
    :return: np.arrary of X_train and y_train
    '''

    save_svs_patches = './data/svs_patches'
    save_viable_patches = './data/viable_patches'
    num_samples = len(samples)

    while 1:
        if shuffle:
            samples = samples.sample(frac=1)
        #select a sub-dataframe with size of batch size
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
                slide_patch, mask_patch = crop(a, (cor1, cor2), (512, 512))[0:2]

                #color normalization
                if color_norm:
                    target = staintools.read_image(target)
                    target = staintools.LuminosityStandardizer.standardize(target)
                    slide_patch = staintools.LuminosityStandardizer.standardize(slide_patch)
                    normalizer = staintools.StainNormalizer(method='vahadane')
                    normalizer.fit(target)
                    slide_patch = normalizer.transform(slide_patch)

                #save patches
                if not os.path.exists(save_svs_patches):
                    os.mkdir(save_svs_patches)
                imsave(osp.join(save_svs_patches, str(a) + '_' + str(cor1) + '_' + str(cor2) + '.png'), slide_patch)
                if not os.path.exists(save_viable_patches):
                    os.mkdir(save_viable_patches)
                imsave(osp.join(save_viable_patches, str(a) + '_' + str(cor1) + '_' + str(cor2) + '_viable.png'), mask_patch)

                images.append(slide_patch)
                masks.append(mask_patch)

                batch_samples = pd.DataFrame(batch_samples)

                X_train = np.array(images)
                y_train = np.array(masks)
                y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], 512, 512, 2)
                print(np.shape(X_train))
                print(np.shape(y_train))

            yield X_train, y_train


def get_start_coordinate(img_id, crop_size):

    '''
    This function is used to get the top left coordinates of all patches which contain at least 50% tissue
    Starting from top left of the whole image, crop a patch at giving size and move with a distance of crop size
    :param img_id: id of the image with the following format: 01_01_0083
    :param crop_size: a tuple stands for the size for each patch
    :return: a dataframe which contains the top left coordinates of all patches which contain at least 50% tissue
    '''
    coordinate = pd.DataFrame([])
    mask_path = './data/ViableMask/' + str(img_id) + '_viable.tif'
    slide_path = './data/OriginalImage/' + str(img_id) + '.svs'
    if not os.path.exists(slide_path):
        slide_path = './data/OriginalImage/' + str(img_id) + '.SVS'
    slide = openslide.open_slide(slide_path)
    mask = io.imread(mask_path)
    mask_shape = np.shape(mask)

    #get all possible y coordinates in the top left coordinates
    img_num_wide = mask_shape[0]//crop_size[0]
    start_y = list(np.arange(img_num_wide) * crop_size[0])
    start_y.append(mask_shape[1]-crop_size[0] + 1)
    #get all possible x coordinates in the top left coordinates
    img_num_height = mask_shape[1]//crop_size[1]
    start_x = list(np.arange(img_num_height) * crop_size[1])
    start_x.append(mask_shape[0] - crop_size[1] + 1)

    for y in start_y:
        for x in start_x:
            croped_slide_img = slide.read_region((x, y), 0, crop_size)
            croped_slide_img = np.array(croped_slide_img)
            # convert the patch from RGBA to grey scale in order to drop the patch which contains much backgrpound
            img_grey = cv2.cvtColor(croped_slide_img, cv2.COLOR_RGBA2GRAY)
            img_grey = np.array(img_grey)
            # drop the coordinates whose patch contained tissue less than 50%
            if np.sum(img_grey>235) > 0.5*512*512:
                coordinate = coordinate.append([[str(x), str(y)]])

    return(coordinate)


def crop(img_id, start, crop_size):

    '''
    :param img_id: id of the image with the following format: 01_01_0083
    :param start: the top left coordinate for the patch
    :param crop_size:  a tuple stands for the size for each patch
    :return: np.array of cropped slide patch, mask patch and top left coordinate of the patch
    '''

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


def gen_imgs_random(id_list, batch_size, crop_size, color_norm=False, target="./data/svs_patches/01_01_0091_12800_22528.png"):

    '''
    :param id_list: a list contains all images ids, all id has the following format: 01_01_0083
    :param batch_size: an int stands for size of the batch
    :param crop_size: a tuple stands for the size for each patch
    :param color_norm: an options whether do the color normalization
    :param target: the path of the base image to do color normalization
    :return: np.arrary of X_train and y_train
    '''

    images = []
    masks = []

    save_svs_patches = './data/svs_patches_random'
    save_viable_patches = './data/viable_patches_random'
    if not os.path.exists(save_svs_patches):
        os.mkdir(save_svs_patches)
    if not os.path.exists(save_viable_patches):
        os.mkdir(save_viable_patches)

    while 1:
        #produce a sample with a fit batch size
        counter = 0
        while counter < batch_size:
            img_id = random.choice(id_list, 1)
            slide_path = './data/OriginalImage/' + str(img_id) + '.svs'
            if not os.path.exists(slide_path):
                slide_path = './data/OriginalImage/' + str(img_id) + '.SVS'
            slide = openslide.open_slide(slide_path)

            #inisilize the top left coordinate for each patch
            shape = np.shape(mask)
            start_x = np.random.randint(shape[0]-crop_size[0])
            start_y = np.random.randint(shape[1]-crop_size[1])

            #if the patch is already cropped, drop this patch
            slide_patch_save_path = osp.join(save_svs_patches, str(img_id) + '_' + str(x) + '_' + str(y) + '.png')
            if not os.path.exists(slide_patch_save_path ):
                croped_slide_img = slide.read_region((start_x, start_y), 0, crop_size)
                croped_slide_img = np.array(croped_slide_img)
                #convert the patch from RGBA to grey scale in order to drop the patch which contains much backgrpound
                img_grey = cv2.cvtColor(croped_slide_img, cv2.COLOR_RGBA2GRAY)
                img_grey = np.array(img_grey)
                #drop the patch where tissue is less than 50%
                if np.sum(img_grey > 235) > 0.5 * 512 * 512:

                    #if option of color normalization is true, do color normalization
                    if color_norm:
                        target = staintools.read_image(target)
                        target = staintools.LuminosityStandardizer.standardize(target)
                        croped_slide_img = staintools.LuminosityStandardizer.standardize(croped_slide_img )
                        normalizer = staintools.StainNormalizer(method='vahadane')
                        normalizer.fit(target)
                        croped_slide_img = normalizer.transform(croped_slide_img )

                    #save patches
                    mask_path = './data/ViableMask/' + str(img_id) + '_viable.tif'
                    mask = io.imread(mask_path)
                    croped_mask_img = mask[start_x:start_x + crop_size[0], start_y:start_y + crop_size[1]]
                    imsave(slide_patch_save_path, croped_slide_img)
                    imsave(osp.join(save_viable_patches, str(a) + '_' + str(cor1) + '_' + str(cor2) + '_viable.png'), croped_mask_img )

                    images.append(slide_patch)
                    masks.append(mask_patch)
                    X_train = np.array(images)
                    y_train = np.array(masks)
                    y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], 512, 512, 2)
                    print(np.shape(X_train))
                    print(np.shape(y_train))

                    counter += 1

        yield X_train, y_train
