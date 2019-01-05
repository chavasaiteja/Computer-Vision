import os
import numpy as np
import cv2


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
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

def SelectiveSearch(image,method,strategy,image_num):
		
	img = cv2.imread(image)	
	cv2.imshow('Window',img)
	cv2.waitKey()	
	
	#Strategies
	stra_color    = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
	stra_texture  = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
	stra_size     = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
	stra_fill     = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
	ss            = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	
	#Feeding an image
	ss.setBaseImage(img)
	
	if method=='q':
		ss.switchToSelectiveSearchQuality()
	elif method=='f':
		ss.switchToSelectiveSearchFast()
	
	
	#Selecting the Strategy:
	if strategy=="color":
		ss.clearStrategies()
		ss.addStrategy(stra_color)
	
	elif strategy == "all":
		stra_multi    = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(stra_texture,stra_color,stra_size,stra_fill)
		ss.clearStrategies()
		ss.addStrategy(stra_multi)
	
	# Get the bounding boxes
	bboxes = ss.process()
	print(len(bboxes))
	lb = len(bboxes)	
	numShowRects = 100	
	count = 0
	while True:
        	# create a copy of original image
		imOut = img.copy()
		imIoU = img.copy()
       		# itereate over all the region proposals
		for i, box in enumerate(bboxes):
			# draw rectangle for region proposal till numShowRects
			if (i < lb):
				x, y, w, h = box
				box1  = [x, y, x+w, y+h] 
				cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
				if image_num == 1:
					for k in gt1:
						iou = bb_intersection_over_union(k,box1)
						if iou>0.5:
							cv2.rectangle(imIoU, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
							count+=1
				elif image_num == 2:
                                        for k in gt2:
                                                iou = bb_intersection_over_union(k,box1)
                                                if iou>0.5:
                                                        cv2.rectangle(imIoU, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                                                        count+=1
				elif image_num == 3:
                                        for k in gt3:
                                                iou = bb_intersection_over_union(k,box1)
                                                if iou>0.5:
                                                        cv2.rectangle(imIoU, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                                                        count+=1
				elif image_num == 4:
                                        for k in gt4:
                                                iou = bb_intersection_over_union(k,box1)
                                                if iou>0.5:
                                                        cv2.rectangle(imIoU, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                                                        count+=1
			else:
                		break
		
		image_name = image+str(image_num)+"with"+".png"
		image_name2= image+str(image_num)+"without"+".png"
        	# show output
		cv2.imshow("Output", imOut)
		cv2.imwrite(image_name,imOut)
		
		print("The count is "+str(count))
		print("% of recall is "+str(count/lb))		

		# Show IoU
		cv2.imshow("IoU",imIoU)
		cv2.imwrite(image_name2,imIoU)

		# record key press
		k = cv2.waitKey(0) & 0xFF
		
        
        	# q is pressed
		if k == 113:
            		break



images = ['HW2_ImageData/HW2_ImageData/Images/2007_000464.jpg', 'HW2_ImageData/HW2_ImageData/Images/2007_001288.jpg', 'HW2_ImageData/HW2_ImageData/Images/2007_002953.jpg','HW2_ImageData/HW2_ImageData/Images/2007_005989.jpg']

gt1 = [[71,252,216,314],[58,202,241,295]]
gt2 = [[68,76,426,209],[357,272,497,317],[196,272,316,314]]
gt3 = [[25,102,93,325],[94,104,166,337],[165,103,247,347],[216,52,416,304],[89,61,162,141]]
gt4 = [[140,130,408,273],[213,96,355,260]]

i = 1
for image in images:
	SelectiveSearch(image,'q',"all",i)
	i+=1

