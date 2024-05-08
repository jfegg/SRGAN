import tqdm
import numpy as np
import glob
from PIL import Image

files = glob.glob('data/lab_data/*.tif')
files2 = glob.glob('data/lab_data_lr/*.tif')

num = 0

for file in tqdm.tqdm(files):
    img = Image.open(file)

    if num < 1000:
        img.save('data/lab_data_hr/image1_' + '{0:03d}'.format(num) + '.tif')
    #print('cellpose_data/gif_masks/' + str(num) + 'mask.gif')
    else:
        img.save('data/lab_data_hr/image2_' + '{0:03d}'.format(num-1000) + '.tif')
    num += 1

num = 0

for file in tqdm.tqdm(files2):
    img = Image.open(file)

    if num < 1000:
        img.save('data/lab_data_lr_2/image1_' + '{0:03d}'.format(num) + '.tif')
    #print('cellpose_data/gif_masks/' + str(num) + 'mask.gif')
    else:
        img.save('data/lab_data_lr_2/image2_' + '{0:03d}'.format(num-1000) + '.tif')
    num += 1