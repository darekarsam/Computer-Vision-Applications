import numpy as np
import imutils
import glob
import cv2
import matplotlib.pyplot as plt


usFlag = cv2.imread('us_flag_color.png')
image = usFlag.copy()

image = cv2.cvtColor(usFlag, cv2.COLOR_BGR2HSV)
mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

image = cv2.Canny(image, 100, 200)
template = cv2.Canny(mask, 50, 200)
(h, w) = template.shape[:2]

# # 
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# for meth in methods:
# 	result = cv2.matchTemplate(image, template, eval(meth))
# 	result2 = np.reshape(result, result.shape[0]*result.shape[1])
# 	sort = np.argsort(result2)
# 	clone = np.dstack([image, image, image])
# 	for i in range(50):
# 		(y, x) = np.unravel_index(sort[i], result.shape) #best match
# 		cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 0, 255), 2)
# 	import ipdb; ipdb.set_trace()
# 	print(meth)
# 	plt.imshow(clone)
# 	plt.show()
# import ipdb; ipdb.set_trace()
# return 0
# for scale in np.linspace(1.0, 2.0, 20):
# resized = imutils.resize(mask, width = int(mask.shape[1] * scale))
# ratio = mask.shape[1] / resized.shape[1]
# edged = cv2.Canny(resized, 50, 200)
# import ipdb; ipdb.set_trace()
result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)#TM_CCOEFF)
result2 = np.reshape(result, result.shape[0]*result.shape[1])
sort = np.argsort(result2)
# (y1, x1) = np.unravel_index(sort[0], result.shape) #best match
# (y2, x2) = np.unravel_index(sort[1], result.shape) #second best match
# (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

clone = np.dstack([image, image, image])
for i in range(50):
	(y, x) = np.unravel_index(sort[i], result.shape) #best match
	cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 0, 255), 2)
# cv2.rectangle(clone, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
# cv2.rectangle(clone, (x2, y2), (x2+w, y2+h), (0, 0, 254), 2)
# cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
# cv2.imshow("Image", clone)
# cv2.waitKey(0)
import ipdb; ipdb.set_trace()
plt.imshow(clone)
plt.show()







# draw a bounding box around the detected region
clone = np.dstack([image, image, image])
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
# cv2.imshow("Visualize", clone)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(clone)
plt.show()
if found is None or maxVal > found[0]:
    found = (maxVal, maxLoc, ratio)

(_, maxLoc, ratio) = found
(startX, startY) = (int(maxLoc[0] * ratio), int(maxLoc[1] * ratio))
(endX, endY) = (int((maxLoc[0] + w) * ratio), int((maxLoc[1] + h) * ratio))

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)



# f=set()
# res = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.6
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#     sensitivity = 1
#     f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))
# cv2.imwrite('res.png',image)
# print(len(f))

# found = None

# for scale in np.linspace(1.0, 2.0, 20):
#     resized = imutils.resize(mask, width = int(mask.shape[1] * scale))
#     ratio = mask.shape[1] / resized.shape[1]
#     edged = cv2.Canny(resized, 50, 200)
#     result = cv2.matchTemplate(image, edged, cv2.TM_CCOEFF)
#     (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

#     (h, w) = resized.shape[:2]

#     # draw a bounding box around the detected region
#     clone = np.dstack([image, image, image])
#     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
#     # cv2.imshow("Visualize", clone)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     plt.imshow(clone)
#     plt.show()
#     if found is None or maxVal > found[0]:
#         found = (maxVal, maxLoc, ratio)

# (_, maxLoc, ratio) = found
# (startX, startY) = (int(maxLoc[0] * ratio), int(maxLoc[1] * ratio))
# (endX, endY) = (int((maxLoc[0] + w) * ratio), int((maxLoc[1] + h) * ratio))

# # draw a bounding box around the detected result and display the image
# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
# cv2.imshow("Image", image)
# cv2.waitKey(0)