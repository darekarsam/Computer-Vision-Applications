import numpy as np
import cv2
import matplotlib.pyplot as plt

def autoCanny(image, sigma=0.33):
	v = np.median(image)

	#---- apply automatic Canny edge detection using the computed median----
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return lower, upper

def readFlag():
	usFlag = cv2.imread('us_flag_color.png')
	return usFlag

def getTemplate():
	mask = cv2.imread('mask.png')
	mask = mask[3:38, 3:40]  #cropping mask
	return mask

def convertHSV(image):
	imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	return imagehsv

def convertLab(image):
	imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
	return imagehsv

def convertGRAY(image):
	imageGRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return imageGRAY

usFlag = readFlag()
usFlagCopy = usFlag.copy()
usFlagHSV = convertHSV(usFlag)

mask = getTemplate()
maskHSV = convertHSV(mask)

lower, upper = autoCanny(usFlagHSV)
image = cv2.Canny(usFlagHSV, lower, upper)

lower, upper = autoCanny(maskHSV)
template = cv2.Canny(maskHSV, lower, upper)
(h, w) = template.shape[:2]

#for all stars except colored ones
result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
result2 = np.reshape(result, result.shape[0]*result.shape[1])
sort = np.argsort(result2)

clone = np.dstack([image, image, image])
for i in range(56):
	# print(i)
	(y, x) = np.unravel_index(sort[i], result.shape) #best match
	cv2.rectangle(usFlagCopy, (x, y), (x+w, y+h), (0, 255, 255), 2)
	cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 255), 2)
	

# for black star
res = cv2.matchTemplate(usFlag,mask,cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(usFlagCopy, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)


#for red star
maskCopy = mask.copy()
maskCopy[np.where((maskCopy==[0,0,0]).all(axis=2))] = [0,255,0]
maskCopyHSV = convertHSV(maskCopy)

res = cv2.matchTemplate(usFlagHSV,maskCopyHSV,cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(usFlagCopy, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)

########

#for Green star
maskCopy = mask.copy()
maskCopy[np.where((maskCopy==[0,0,0]).all(axis=2))] = [255,0,255]
maskCopyLab = convertLab(maskCopy)

res = cv2.matchTemplate(usFlag, maskCopyLab, cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(usFlagCopy, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)



#for Pink
maskCopy = mask.copy()
maskCopy[np.where((maskCopy==[0,0,0]).all(axis=2))] = [213,106,252]
maskCopy[np.where((maskCopy==[255,255,255]).all(axis=2))] = [110,59,60]
maskCopyHSV = convertHSV(maskCopy)

res = cv2.matchTemplate(usFlag, maskCopyHSV, cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
cv2.rectangle(usFlagCopy, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)
cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 255), 2)

cv2.imwrite('EdgeOutput.png',clone)
cv2.imwrite('finalOutput.png',usFlagCopy)