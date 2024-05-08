import tqdm
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# files = glob.glob('./data/hr1/*.tif')
# files2 = glob.glob('./data/lr1/*.tif')

files = os.listdir('./data/hr1')
files2 = os.listdir('./data/lr1')

files.sort()
files2.sort()

for i, file in enumerate(files):
    
    img_hr = files[i]
    img_lr = files2[i]

    img_hr = cv2.imread('./data/hr1/' + img_hr)
    img_lr = cv2.imread('./data/lr1/' + img_lr)

    ##8 bit conversion
    img_hr = img_hr.astype(np.float16) / img_hr.max()

    img_hr *= 2**16

    img_hr = (img_hr/256).astype('uint8')

    img_lr = img_lr.astype(np.float16) / img_lr.max()

    img_lr *= 2**16

    img_lr = (img_lr/256).astype('uint8')

    x, y, _ = img_hr.shape
    # img_hr_resized = cv2.resize(img_hr, (x, y))
    img_lr_resized = cv2.resize(img_lr, (x, y))
    ##8 bit conversion

    ##messing with the images for open cv keypoints

    img_hr_org = img_hr
    img_lr_org = img_lr

    img_hr[img_hr > 10] = 255
    img_lr[img_lr > 5] = 255
    # img_hr_resized[img_hr_resized > 10] = 255
    img_lr_resized[img_lr_resized > 10] = 255

    img_hr_padded = cv2.copyMakeBorder(img_hr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, None, value = 0)

    res = cv2.matchTemplate(img_lr_resized, img_hr_padded, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    x_shift = top_left[0] - 30
    y_shift = top_left[1] - 30

    transformation = np.float32([[1, 0, x_shift/4], [0, 1, y_shift/4]])

    img_lr_translated = cv2.warpAffine(img_lr_org, transformation, (256, 256))


    img_lr_translated = cv2.cvtColor(img_lr_translated, cv2.COLOR_BGR2GRAY)
    img_hr = cv2.cvtColor(img_hr_org, cv2.COLOR_BGR2GRAY)

    # img_hr[img_hr > 10] = 255
    # img_lr_translated[img_lr_translated > 5] = 255

    if i < 200:
        cv2.imwrite("./data/lr_train/lr_" + str(i) +".png", img_lr_translated)
        cv2.imwrite("./data/hr_train/hr_" + str(i) +".png", img_hr)
    elif i < 300:
        cv2.imwrite("./data/lr_test/img_" + str(i) +".png", img_lr_translated)
        cv2.imwrite("./data/hr_test/img_" + str(i) +".png", img_hr)
    else:
        exit()

