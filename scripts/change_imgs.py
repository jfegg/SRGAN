import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

files = glob.glob('./data/density50/*.tif')

for i, file in enumerate(files):
    
    img = files[i]

    img_hr = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    img_hr = img_hr.astype(np.float16) / img_hr.max()

    img_hr *= 2**16

    img_hr = (img_hr/256).astype('uint8')

    cv2.imwrite("./data/density50_mod/img_" + str(i) +".png", img_hr)



