import matplotlib
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from skimage import io, transform
import scipy.misc as misc
from pydaily import filesystem
from pyslide import pyramid
import openslide
from PIL import Image
from skimage import io

def get_id_list(slides_dir):
    id_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    id_list.extend([os.path.basename(ele) for ele in svs_file_list])
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    id_list.extend([os.path.basename(ele) for ele in SVS_file_list])
    id = [os.path.splitext(ele)[0] for ele in id_list]
    return id


def slide_combine_mask(slides_dir, id_list, slide_index, display_level=2):
    """
    Load slide segmentation mask.
    """

    slide_path = os.path.join(slides_dir, 'OriginalImage/' + id_list[slide_index] + ".svs")
    if not os.path.exists(slide_path):
        slide_path = os.path.join(slides_dir, 'OriginalImage/' + id_list[slide_index] + ".SVS")

    wsi_head = pyramid.load_wsi_head(slide_path)
    new_size = (wsi_head.level_dimensions[display_level][1], wsi_head.level_dimensions[display_level][0])
    slide_img = wsi_head.read_region((0, 0), display_level, wsi_head.level_dimensions[display_level])
    slide_img = np.asarray(slide_img)[:,:,:3]
    # lo
    # ad and resize whole mask
    whole_mask_path = os.path.join(slides_dir, 'WholeMask/' + id_list[slide_index] + "_whole.tif")
    whole_mask_img = io.imread(whole_mask_path)
    resize_whole_mask = (transform.resize(whole_mask_img, new_size, order=0) * 255).astype(np.uint8)
    # load and resize viable mask
    viable_mask_path = os.path.join(slides_dir, 'ViableMask/' + id_list[slide_index]+"_viable.tif")
    viable_mask_img = io.imread(viable_mask_path)
    resize_viable_mask = (transform.resize(viable_mask_img, new_size, order=0) * 255).astype(np.uint8)

    # show the mask
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    ax1.imshow(slide_img)
    ax1.set_title('Slide Image')
    ax2.imshow(resize_whole_mask)
    ax2.set_title('Whole Tumor Mask')
    ax3.imshow(resize_viable_mask)
    ax3.set_title('Viable Tumor Mask')
    plt.tight_layout()
    plt.show()

    dir_path = './data/MyMasks_level_' + str(display_level)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    save_path = os.path.join(dir_path, 'level_' + str(display_level) + '_' + id_list[slide_index] + ".png")
    fig.savefig(save_path)


def save_masks(slides_dir, id_list, display_level=2):
    slide_num = len(id_list)
    for ind in np.arange(slide_num):
        print("processing {}/{}".format(ind+1, slide_num))
        slide_combine_mask(slides_dir, id_list, ind, display_level=display_level)

if __name__ ==  '__main__':
    id_list = get_id_list('./data/OriginalImage')
    for i in len(id_list):
        slide_combine_mask('./data/My_masks_level_1', id_list, i, display_level=1):






