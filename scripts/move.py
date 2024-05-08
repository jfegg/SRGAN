import tqdm
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2

files = glob.glob('./data/hr_train/*.png')
files2 = glob.glob('./data/lr_train/*.png')

for i, file in tqdm.tqdm(enumerate(files)):
    if i < 250:
        os.remove(files[i])
        os.remove(files2[i])

