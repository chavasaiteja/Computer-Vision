#from __future__ import print_function

import os
import numpy as np
import cv2

def mean_shift_segmentor(img_filename, spatial_radius, color_radius):
	src_img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
	img_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab)

	# Find the peak of a color-spatial distribution
	# pyrMeanShiftFiltering(src, spatialRadius, colorRadius, max_level)
	# For 640x480 color image, it works well to set spatialRadius equal to 2 and colorRadius equal to 40
	# max_level describes how many levels of scale pyramid you want to use for segmentation
	# A max_level of 2 or 3 works well for a 640x480 color image
	dst = cv2.pyrMeanShiftFiltering(img_lab, spatial_radius, color_radius, 1)
	dst = cv2.cvtColor(dst, cv2.COLOR_Lab2BGR)

	# filename
	dst_filename = os.path.splitext(img_filename)[0] + '_meanshift_spatial_' + str(spatial_radius) + '_color_' + str(color_radius) + os.path.splitext(img_filename)[1]
	print('dst_filename: ' + dst_filename)

	cv2.imwrite(dst_filename, dst)

	return dst

images = ['HW2_ImageData/HW2_ImageData/Images/2007_000464.jpg', 'HW2_ImageData/HW2_ImageData/Images/2007_001288.jpg', 'HW2_ImageData/HW2_ImageData/Images/2007_002953.jpg','HW2_ImageData/HW2_ImageData/Images/2007_005989.jpg']

for image in images:
	color_radius = 15
	for spatial_radius in range(15, 60, 15):
		mean_shift_segmentor(image, spatial_radius, color_radius)
	
	spatial_radius = 15
	for color_radius in range(15,60,15):
		mean_shift_segmentor(image, spatial_radius, color_radius)

	
			
			


