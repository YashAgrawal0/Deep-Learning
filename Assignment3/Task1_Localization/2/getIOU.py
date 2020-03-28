import numpy as np

ip_folder = "./results"
output_folder = ip_folder + "/IOU.txt"
y_pred = np.load(ip_folder + "/y_pred.npy")
y_test = np.load(ip_folder + "/y_test.npy")

print (y_pred.shape, y_test.shape)

def bb_intersection_over_union(boxA, boxB):
	
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	if((xB - xA)==0 or (yB - yA)==0):
		iou = 0.0
		return iou
	interArea =  max(0, xB - xA + 1) * max(0, yB - yA + 1)                                                                                                                                                                                                                                                             
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

iou = []
length = len(y_pred)

for i in range(length):
	value = 0
	value = bb_intersection_over_union(y_pred[i][0:4], y_test[i][0:4])
	value += bb_intersection_over_union(y_pred[i][4:8], y_test[i][4:8])
	value += bb_intersection_over_union(y_pred[i][8:12], y_test[i][8:12])
	value += bb_intersection_over_union(y_pred[i][12:16], y_test[i][12:16])
	iou.append(value)

print("Writing to file: " + output_folder)
with open(output_folder, 'w') as f:
    for data in iou:
        f.write("%s\n" % data)
print("Done!")
