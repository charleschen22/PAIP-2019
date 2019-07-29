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
from PIL import Image
import cv2
from keras.utils.np_utils import to_categorical


print('Patches producing can take a while...')
#slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'
#BASE_TRUTH_DIR = '/home/wli/Downloads/CAMELYON16/masking'
svs_dir_ = './data/OriginalImage'
viable_dir_ = './data/ViableMask'
whole_dir_ = './data/WholeMask'

#slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
svs_paths = glob.glob(osp.join(svs_dir_, '*.svs'))
svs_paths.extend(glob.glob(osp.join(svs_dir_, '*.SVS')))
viable_paths = glob.glob(osp.join(viable_dir_, '*.tif'))
whole_paths = glob.glob(osp.join(whole_dir_, '*tif'))

#slide_paths.sort()
#BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
#BASE_TRUTH_DIRS.sort()
# image_pair = zip(tumor_paths, anno_tumor_paths)
# image_pair = list(image_mask_pair)
patch_path = '/home/wli/Documents/patches'
sample_total = pd.DataFrame([])
i = 0
while i < len(svs_paths):

    viable_dir = Path(viable_dir_)

    with openslide.open_slide(svs_paths[i]) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 512, slide.dimensions[1] / 512))

        thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale
        thresh = threshold_otsu(thumbnail_grey)
        binary = thumbnail_grey > thresh

        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = ~patches[0]
        patches.drop(0, axis=1, inplace=True)
        patches['slide_path'] = svs_paths[i]
 
    # if filter_non_tissue:
    samples = patches
    samples = samples[samples.is_tissue == True]  # remove patches with no tissue
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)

    sample_total = sample_total.append(samples, ignore_index=True)

    i = i + 1

NUM_CLASSES = 2  # not_tumor, tumor


def gen_imgs(samples, batch_size, base_truth_dir, shuffle=True):

    save_svs_patches = './data/svs_patches'
    save_viable_patches = './data/viable_patches'
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1)  # shuffle samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            # images = []
            # masks = []
            for _, batch_sample in batch_samples.iterrows():
                #slide_contains_tumor = osp.basename(batch_sample.slide_path).startswith('tumor_')

                with openslide.open_slide(batch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
                    img = tiles.get_tile(tiles.level_count - 1, batch_sample.tile_loc[::-1])
                    img = np.array(img)
                    cor1, cor2 = batch_sample.tile_loc[::-1]

                    if not os.path.exists(save_svs_patches):
                        os.mkdir(save_svs_patches)
                    id = osp.splitext(osp.basename(batch_sample.slide_path))[0]
                    imsave(osp.join(save_svs_patches, str(id) + '_' + str(cor1) + '_' + str(cor2) + '.png'), img)

                # only load truth mask for tumor slides
                viable_path = './data/ViableMask' + str(id) + '_viable.tif'
                with openslide.open_slide(viable_path) as mask:
                        truth_tiles = DeepZoomGenerator(mask, tile_size=512, overlap=0, limit_bounds=False)
                        mask = truth_tiles.get_tile(truth_tiles.level_count - 1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        mask = np.array(mask)
                        cor1, cor2 = batch_sample.tile_loc[::-1]

                    if not os.path.exists(save_viable_patches):
                        os.mkdir(save_viable_patches)
                    imsave(osp.join(save_viable_patches, str(id) + '_' + str(cor1) + '_' + str(cor2) + '_viable.png'), mask)

            yield

