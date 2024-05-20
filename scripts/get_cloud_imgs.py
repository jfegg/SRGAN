from cloudvolume import CloudVolume
import random
import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch2', parallel=True, progress=True)




for i in range(20):
    z = random.randint(400,1200)
    image = vol[:, :, z]

    x_lim = image.shape[0]
    y_lim = image.shape[1]

    for i in range(50):
        x = random.randint(1, (x_lim / 2000)) * 2000
        y = random.randint(1, (y_lim / 2000)) * 2000


        img_section = image[x-1000:x, y-1000:y, 0, 0]

        img_section = img_section.astype(np.float16) / img_section.max()

        img_section *= 2**8

        img_section = (img_section).astype('uint8')

        cv2.imwrite("./data/real_hr_train/test_" + str(x) + "_" + str(y) + "_" + str(z) +".png", img_section)

        img_lr = cv2.resize(img_section, (int(img_section.shape[0] / 4), int(img_section.shape[1] / 4)))

        cv2.imwrite("./data/real_lr_train/test_" + str(x) + "_" + str(y) + "_" + str(z) +".png", img_lr)



