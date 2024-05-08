import tqdm
import numpy as np
import glob
from PIL import Image
import cv2

files = glob.glob('./data/art_mod/*.png')

for i, file in enumerate(files):
    
    img = files[i]

    gt_image = cv2.imread(img).astype(np.float32) / 255.
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

    cv2.imshow("img", gt_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


