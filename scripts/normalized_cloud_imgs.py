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

imgs = []

while count < 100:
    z = random.randint(300,1000)
    x = random.randint(1, (x_lim / 2000)) * 2000
    y = random.randint(1, (y_lim / 2000)) * 2000


    img = vol[x-1000:x, y-1000:y, z]
    img = img[:, :, 0, 0]

    if np.std(img) < 17:
        continue

    count += 1

    imgs.append(img)

    print(count) 

percentiles = []
for i in range(len(imgs)):
    percentiles.append(np.percentile(imgs[i], 99.99))

percentiles = np.array(percentiles)

percentile = percentiles.mean()

print(percentile)

for i in range(len(imgs)):

    img = imgs[i]

    img = img.astype(np.float16) / 360

    img[img > 1] = 1

    img *= 255

    img = (img).astype('uint8')

    cv2.imwrite("./data/real_hr_train_normal/test_" + str(i) + ".png", img)

    img_lr = cv2.resize(img, (int(img.shape[0] / 4), int(img.shape[1] / 4)))

    cv2.imwrite("./data/real_lr_train_normal/test_" + str(i) +".png", img_lr)



