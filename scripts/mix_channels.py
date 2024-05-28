import tqdm
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2

files = glob.glob('./data/real_train_splices/*.png')
files2 = glob.glob('./data/real_train_ch0_splices/*.png')

for i, file in tqdm.tqdm(enumerate(files)):
    img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
    cv2.imwrite("./data/real_train_ch0and2_splices/splice_ch2_" + str(i) + ".png", img)


for i, file in tqdm.tqdm(enumerate(files2)):
    img = cv2.imread(files2[i], cv2.IMREAD_UNCHANGED)
    cv2.imwrite("./data/real_train_ch0and2_splices/splice_ch0_" + str(i) + ".png", img)
