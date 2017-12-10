import numpy as np
import imutils
import glob
import cv2
import matplotlib.pyplot as plt

usFlag = cv2.imread('us_flag_color.png')
img = usFlag.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.imread('mask.png')
maskC = mask[3:38, 3:40]
template = maskC.copy()
template[np.where((template==[0,0,0]).all(axis=2))] = [0,255,0]
# import ipdb; ipdb.set_trace()
template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(h, w) = template.shape[:2]
plt.subplot(1,2,1),plt.imshow(img)
plt.title('flag')
plt.subplot(1,2,2),plt.imshow(template)
plt.title('template')
plt.show()

res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.4627
# loc = np.where(res>=threshold)

# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)

(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(usFlag, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)

cv2.imshow("detected",usFlag)
cv2.waitKey(0)
# cv2.imwrite('img_bgr.png',img_bgr)
# cv2.imwrite('gray_image.png',img_gray)
###############################
# convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

# import ipdb; ipdb.set_trace()

# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()

# sob = image.c

# result = cv2.matchTemplate(sobelx, template, cv2.TM_SQDIFF_NORMED)
# (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
# clone = np.dstack([image, image, image])
# cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
# plt.imshow(clone)
# plt.show()


# import ipdb; ipdb.set_trace()
############################
# clone = np.dstack([image, image, image])
# threshold = 10000000
# 			# 9753750
# loc = np.where( result >= threshold)
# f = set()

# for pt in zip(*loc[::-1]):
#     cv2.rectangle(clone, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#     sensitivity = 150
#     f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))

# cv2.imwrite('resthresh.png',clone)

# print(len(f))
# x 505  y 345
##############################