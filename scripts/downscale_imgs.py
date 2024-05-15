import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

files1 = glob.glob('./data/density1_test_mod/*.png')
files2 = glob.glob('./data/density5_test_mod/*.png')
files3 = glob.glob('./data/density10_test_mod/*.png')
files4 = glob.glob('./data/density50_test_mod/*.png')

for i in range(5):
    
    img1 = files1[i]
    img2 = files2[i]
    img3 = files3[i]
    img4 = files4[i]

    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img3 = Image.open(img3)
    img4 = Image.open(img4)

    img1_lr = img1.resize((int(img1.size[0] * 0.25), int(img1.size[1] * 0.25)))
    img2_lr = img2.resize((int(img1.size[0] * 0.25), int(img1.size[1] * 0.25)))
    img3_lr = img3.resize((int(img1.size[0] * 0.25), int(img1.size[1] * 0.25)))
    img4_lr = img4.resize((int(img1.size[0] * 0.25), int(img1.size[1] * 0.25)))

    img1.save("./data/art_hr_test/img_1" + str(i) +".png")
    img2.save("./data/art_hr_test/img_2" + str(i) +".png")
    img3.save("./data/art_hr_test/img_3" + str(i) +".png")
    img4.save("./data/art_hr_test/img_4" + str(i) +".png")

    img1_lr.save("./data/art_lr_test/img_1" + str(i) +".png")
    img2_lr.save("./data/art_lr_test/img_2" + str(i) +".png")
    img3_lr.save("./data/art_lr_test/img_3" + str(i) +".png")
    img4_lr.save("./data/art_lr_test/img_4" + str(i) +".png")



