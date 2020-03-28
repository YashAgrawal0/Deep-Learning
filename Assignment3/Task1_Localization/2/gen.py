import numpy as np
import os
import cv2
# import imutils

output_classes = 16
folder = "/users/home/dlagroup5/wd/Assignment3/data/Q1/Four_Slap_Fingerprint"
output_folder = "/users/home/dlagroup5/wd/Assignment3/Task1_Localization/Task2/input"
GT_folder = "Ground_truth"
image_folder = "Image"
training_samples = 0.7
testing_samples = 1 - training_samples
filename = "3430_seg"


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def flipv(imgg):
 img2= np.zeros([r, c, 3], np.uint8)
 for i in range(r): img2[i,:]=imgg[r-i-1,:]
 return img2

def getData(lower, upper):
	image_dir = folder + "/" + image_folder
	filenames = []

	for filename in os.listdir(image_dir):
		filename = filename.split(".")[0]
		filenames.append(filename)

	TOTAL_FILES = len(filenames)

	Y = []
	images = []
	L, U = int(TOTAL_FILES * lower), int(TOTAL_FILES * upper)
	for i in range(L, U):
		filename = filenames[i]
		print(i, "Processing: ", filename)
		img = cv2.imread(folder + "/" + image_folder + "/" + filename + ".jpg")
		img = rotateImage(img, 90)
		global r, c
		r, c = img.shape[:2]
		img = flipv(img)

		# [X translation],[Y translation]
		T = np.float32([[1,0,-50],[0,1,50]])
		img=cv2.warpAffine(img,T,(c,r))

		f = open(folder + "/" + GT_folder + "/" + filename + ".txt")
		y_single_img = []
		for line in f:
			y_ = [int(x) for x in line.split(",")]
			for i in y_:
				y_single_img.append(i)

		Y.append(y_single_img)
		images.append(img)

		# r, c = img.shape[:2]
		# print (Y.tolist())
		# print (r, c)
		# for i in range(4):
		# 	rect = cv2.rectangle(img, (y_single_img[i*4 + 0], y_single_img[i*4 + 1]), (y_single_img[i*4 + 2], y_single_img[i*4 + 3]), (0, 255, 0), 2)
		# cv2.imwrite("./augmented/" + filename + ".jpg", rect)

	Y = np.array(Y)
	images = np.array(images)

	return images,Y

X_train, y_train = getData(0, training_samples)
X_test, y_test = getData(training_samples, 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

np.save(output_folder + "/X_train.npy", X_train)
np.save(output_folder + "/y_train.npy", y_train)
np.save(output_folder + "/X_test.npy", X_test)
np.save(output_folder + "/y_test.npy", y_test)
