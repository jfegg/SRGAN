import tqdm
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2


files = glob.glob('./data/hr1/*.tif')
files2 = glob.glob('./data/lr1/*.tif')

img_hr = files[505]
img_lr = files2[505]

img_hr = cv2.imread(img_hr)
img_lr = cv2.imread(img_lr)

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

img_hr[img_hr > 10] = 255
img_lr[img_lr > 5] = 255
# img_hr_resized[img_hr_resized > 10] = 255
img_lr_resized[img_lr_resized > 10] = 255

img_hr_padded = cv2.copyMakeBorder(img_hr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, None, value = 0)

# cv2.imshow("HR", img_hr)
# cv2.imshow("LR", img_lr)
# cv2.imshow("LR resized", img_lr_resized)
# cv2.imshow("HR padded", img_hr_padded)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


res = cv2.matchTemplate(img_lr_resized, img_hr_padded, cv2.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)



plt.imshow(res,cmap = 'gray')
plt.savefig('fig.png')


top_left = max_loc

print(top_left)

x_shift = top_left[0] - 30
y_shift = top_left[1] - 30

print(x_shift)
print(y_shift)

transformation = np.float32([[1, 0, x_shift/4], [0, 1, y_shift/4]])

img_lr_translated = cv2.warpAffine(img_lr, transformation, (256, 256))

# cv2.imshow("Translation", img_lr_translated)
# # cv2.imshow("Untranslated", img_lr_resized)
# cv2.imshow("HR", img_hr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_lr_translated = cv2.cvtColor(img_lr_translated, cv2.COLOR_BGR2GRAY)
img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)

img_hr[img_hr > 10] = 255
img_lr_translated[img_lr_translated > 5] = 255

cv2.imwrite("./scripts/translated/translated.png", img_lr_translated)
cv2.imwrite("./scripts/translated/normal_hr.png", img_hr)

