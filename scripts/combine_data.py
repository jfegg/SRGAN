import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

files1 = glob.glob('./data/density1_mod/*.png')
files2 = glob.glob('./data/density5_mod/*.png')
files3 = glob.glob('./data/density10_mod/*.png')
files4 = glob.glob('./data/density50_mod/*.png')

for i in range(200):
    
    img1 = files1[i]
    img2 = files2[i]
    img3 = files3[i]
    img4 = files4[i]

    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img3 = Image.open(img3)
    img4 = Image.open(img4)

    img1.save("./data/mixed_mod/img_1" + str(i) +".png")
    img2.save("./data/mixed_mod/img_2" + str(i) +".png")
    img3.save("./data/mixed_mod/img_3" + str(i) +".png")
    img4.save("./data/mixed_mod/img_4" + str(i) +".png")



