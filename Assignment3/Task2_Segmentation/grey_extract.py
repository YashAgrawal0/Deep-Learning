import cv2
import os
input_folder = "/users/home/dlagroup5/wd/Assignment3/data/Q2/"
x = cv2.imread(input_folder+"a.tiff")
# print(x.shape)
cv2.imwrite('img1.jpg',x)
# cv2.waitKey(0)