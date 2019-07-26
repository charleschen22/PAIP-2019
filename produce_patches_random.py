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


def get_id_list(slides_dir):
    id_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    id_list.extend([os.path.basename(ele) for ele in svs_file_list])
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    id_list.extend([os.path.basename(ele) for ele in SVS_file_list])
    id = [os.path.splitext(ele)[0] for ele in id_list]
    return id


id_list = get_id_list('./data/OriginalImage')


def sequential_crop(id, start, crop_size):

    start_x, start_y = start
    slide_path = './data/OriginalImage/' + str(id) + '.svs'
    if not os.path.exists(slide_path):
        slide_path = openslide.open_slide('./data/OriginalImage/' + str(id) + '.SVS')
    slide = openslide.open_slide(slide_path)
    croped_slide_img = slide.read_region((start_x, start_y), 0, crop_size)
    croped_slide_img = np.array(croped_slide_img)
    mask_path = './data/ViableMask/' + str(id) + '_viable.tif'
    mask = io.imread(mask_path)
    croped_mask_img = mask[start_x:start_x+crop_size[1], start_y:start_y+crop_size[0]]

    return (croped_slide_img, croped_mask_img, start)


def get_start_coordinate(id, crop_size):
    coordinate = pd.DataFrame([])
    mask_path = './data/ViableMask/' + str(id) + '_viable.tif'
    mask = io.imread(mask_path)
    mask_shape = np.shape(mask)
    img_num_wide = mask_shape[0]//crop_size[1]
    img_num_height = mask_shape[1]//crop_size[0]
    for i in range(img_num_height):
        for j in range(img_num_wide):
            x = j*crop_size[1]
            y = i*crop_size[0]
            coordinate = coordinate.append([[[x, y]]])
    coordinate = coordinate.append([[[mask_shape[0]-crop_size[1]+1, mask_shape[1]-crop_size[0]+1]]])

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

patch_path = '/home/wli/Documents/patches'
sample_total = pd.DataFrame([])
i = 0


while i < len(id_list):
    print('Processing data #' + str(i) + '............')
    patches = get_start_coordinate(id_list[i], (512, 512))
    patches['id'] = id_list[i]
    sample_total = sample_total.append(patches, ignore_index=True)

    i += 1

sample_total.to_csv('./data/coordinates.csv')

def gen_imgs(samples, batch_size, shuffle=True):

    save_svs_patches = './data/svs_patches'
    save_viable_patches = './data/viable_patches'
    num_samples = len(samples)
    while 1:
        if shuffle:
            samples = samples.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            for i in range(batch_size):
                id = batch_samples[i, 1]
                cor = batch_samples[i, 0]
                cor1, cor2 = cor

                slide_patch = sequential_crop(id, cor, (512, 512))
                mask_patch = sequential_crop(id, cor, (512, 512))

                if not os.path.exists(save_svs_patches):
                    os.mkdir(save_svs_patches)
                imsave(osp.join(save_svs_patches, str(id) + '_' + str(cor1) + '_' + str(cor2) + '.png'), slide_patch)
                if not os.path.exists(save_viable_patches):
                    os.mkdir(save_viable_patches)
                imsave(osp.join(save_viable_patches, str(id) + '_' + str(cor1) + '_' + str(cor2) + '_viable.png'), mask_patch)

            yield

gen_imgs(smaple_total, 32)