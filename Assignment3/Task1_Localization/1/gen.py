import numpy as np
import os
import cv2

# Class 0 = knuckle
# Class 1 = Palm
# Class 2 = veins
output_classes = 3
output_regression = 4
# folder = "../../../data/Q1"
folder = "/users/home/dlagroup5/wd/Assignment3/data/Q1"
class_folders = ["Knuckle", "Palm", "Vein"]
classes = ["knuckle", "Palm", "veins"]
# output_folder = "../input"
output_folder = "/users/home/dlagroup5/wd/Assignment3/Task1_Localization/Task1/input"
GT = "groundtruth.txt"
training_samples = 0.7
testing_samples = 1 - training_samples
R, C = 480, 640

def getData(lower, upper):
    X, y_classification, y_regression = [], [], []
    for i in range(output_classes):
        print ("Class " + str(i) + "->"),
        lines = open(folder + "/" + class_folders[i] + "/" + GT, "r").readlines()
        sz = len(lines)
        L = int(lower * sz)
        U = int(upper * sz)
        for j in range(L, U):
            pic_path, x1, y1, x2, y2, cl = lines[j].split(",")
            cl = classes.index(cl[:-1]) # for removing '\n' from the end of the word
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = cv2.imread(folder + "/" + class_folders[i] + "/" + pic_path)
            if(j % 300 == 0): print(" " + str(j)),
            if(img is not None):
                r, c = img.shape[:2]
    #             print ("Original Shape: "),
    #             print(img.shape),
                img = cv2.copyMakeBorder(img, top=0, left=0, right=C - c, bottom=R - r, borderType= cv2.BORDER_CONSTANT)
    #             print (" After padding Shape: "),
    #             print(img.shape)
                X.append(img)
                y_classification.append([0] * output_classes)
                y_regression.append([x1, y1, x2, y2])
                y_classification[i * (U - L) + j - L][cl] = 1
    #         rect = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.imshow("image", rect)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
        print("\n")
    X = np.array(X)
    y_classification = np.array(y_classification)
    y_regression = np.array(y_regression)
    return X, y_classification, y_regression
            
X_train, y1_train, y2_train = getData(0, training_samples)

X_test, y1_test, y2_test = getData(training_samples, 1)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
np.save(output_folder + "/X_train.npy", X_train)
np.save(output_folder + "/y1_train.npy", y1_train)
np.save(output_folder + "/y2_train.npy", y2_train)
np.save(output_folder + "/X_test.npy", X_test)
np.save(output_folder + "/y1_test.npy", y1_test)
np.save(output_folder + "/y2_test.npy", y2_test)
