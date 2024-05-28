from cloudvolume import CloudVolume
import random
import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

x_lim = 28000
y_lim = 28000
count = 0

while count < 1024:
    z = random.randint(300,1000)
    x = random.randint(1, (x_lim / 2000)) * 2000
    y = random.randint(1, (y_lim / 2000)) * 2000


    img = vol[x-1000:x, y-1000:y, z]
    img = img[:, :, 0, 0]

    if np.std(img) < 17:
        continue

    count += 1

    print(count) 

    img = img.astype(np.float16) / img.max()

    img *= 2**8

    img = (img).astype('uint8')

    cv2.imwrite("./data/real_hr_train_ch0/test_" + str(x) + "_" + str(y) + "_" + str(z) + ".png", img)

    img_lr = cv2.resize(img, (int(img.shape[0] / 4), int(img.shape[1] / 4)))

    cv2.imwrite("./data/real_lr_train_ch0/test_" + str(x) + "_" + str(y) + "_" + str(z) +".png", img_lr)



