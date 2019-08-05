
# coding: utf-8

# In[ ]:

from skimage import io
from keras.models import load_model
import numpy as np
import cv2

model = load_model('./model/random_model_unet.h5')

img = io.imread('./data/svs_patches_random/01_01_0085_28518_19435.png')
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
print(img)
img = np.resize(img, (1, 512, 512, 4))
print(np.shape(img))
mask = io.imread('./data/ViableMask/01_01_0085_viable.tif')
mask = mask[28518:28518+512, 19435:19435+512]

pre = model.predict(img)
pre = -np.argmax(pre, axis=3)+np.ones((512, 512))
print(pre)
print(mask)
print(np.sum(pre==mask)/512/512)

