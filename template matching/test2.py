import numpy as np
import cv2
import matplotlib.pyplot as plt


usFlag = cv2.imread('us_flag_color.png')
image1 = usFlag.copy()

image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
imagehsv = image.copy()

mask = cv2.imread('mask.png')

maskCropped = mask[3:38, 3:40]  #cropping mask
maskCopy = maskCropped.copy()
maskC = cv2.cvtColor(maskCropped, cv2.COLOR_BGR2HSV)

# maskc = cv2.GaussianBlur(maskC,(3,3),0)
# maskC = cv2.cvtColor(maskC, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('maskHSV.png',maskC)
#crop x=3,y=3, x1=40 ,y1=38
# import ipdb; ipdb.set_trace()
# image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)

def autoCanny(image, sigma=0.33):
	v = np.median(image)

	#---- apply automatic Canny edge detection using the computed median----
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return lower, upper
lower, upper = autoCanny(image)
# import ipdb; ipdb.set_trace()
image = cv2.Canny(image, lower, upper)
# cv2.imwrite('flagHSV.png',image)

lower, upper = autoCanny(maskC)
print(lower, upper)
template = cv2.Canny(maskC, lower, upper)
(h, w) = template.shape[:2]
# plt.imshow(template)
# plt.show()

plt.subplot(1,2,1),plt.imshow(image)
plt.title('flag')
plt.subplot(1,2,2),plt.imshow(template)
plt.title('template')
plt.show()
import ipdb; ipdb.set_trace()

result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
result2 = np.reshape(result, result.shape[0]*result.shape[1])
sort = np.argsort(result2)

clone = np.dstack([image, image, image])
for i in range(111):
	# print(i)
	(y, x) = np.unravel_index(sort[i], result.shape) #best match
	cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 0, 255), 2)
	cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 0, 255), 2)
	# import ipdb; ipdb.set_trace()

#for black image
res = cv2.matchTemplate(usFlag,mask,cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(image1, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)




maskCopy[np.where((maskCopy==[0,0,0]).all(axis=2))] = [0,255,0]
maskCopy = cv2.cvtColor(maskCopy, cv2.COLOR_BGR2HSV)

plt.subplot(1,2,1),plt.imshow(imagehsv)
plt.title('flag')
plt.subplot(1,2,2),plt.imshow(maskCopy)
plt.title('template')
plt.show()

res = cv2.matchTemplate(imagehsv,maskCopy,cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(image1, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 253, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 253, 255), 2)


cv2.imwrite('res.png',image1)
cv2.imwrite('resClone.png',clone)
# plt.imshow(clone)
# plt.show()