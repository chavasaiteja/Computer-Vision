import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--detect", required = True, type=str, help = "path to detect image")
ap.add_argument("-t","--target", required = True, type=str, help = "path to target image")
args = vars(ap.parse_args())

img_detect = args["detect"]
img_target = args["target"]

# Read the images.
img1   = cv2.imread(img_detect)
img2   = cv2.imread(img_target)
image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # queryImage
image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # trainImage

#Display the detect image
cv2.imshow('Detect',image1)
cv2.waitKey(0)

#Display the target image
cv2.imshow('Target',image2)
cv2.waitKey(0)

# Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()
#sift =  cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

print('The features found in querry image ' + img_detect + ': ' + str(des1.shape[0]))
print('The features found in train image ' + img_target + ': ' + str(des2.shape[0]))

# features overlaid on the images with circles based on size of keypoints and its orientation
img4 = cv2.drawKeypoints(image1,kp1,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv2.drawKeypoints(image2,kp2,image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


#Show the query image with SIFT keypoints
cv2.imshow('Query image with SIFT keypoints',img4)
cv2.waitKey(0)
s1 = img_detect.split(".")
cv2.imwrite(s1[0] +' with SIFT keypoints'+'.png',img4)

#Display the train image with SIFT keypoints
cv2.imshow('Train image with SIFT keypoints',img5)
cv2.waitKey(0)
s2 = img_target.split(".")
cv2.imwrite(s2[0] +' with SIFT keypoints'+'.png',img5)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
# store all the good matches as per Lowe's ratio test.
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

# Sort the good matches in the order of their distance and plot the top 20 matches
good_matches = sorted(good_matches, key = lambda x:x.distance)
img6 = cv2.drawMatches(image1, kp1, image2, kp2, good_matches[:20], None, flags=2)
cv2.imshow('Top 20 matches between ' + img_detect + ' and ' + img_target, img6)
cv2.waitKey(0)
s3 = img_detect.split('/')
s4 = img_target.split('/')
cv2.imwrite('HW3_Data/'+'Top 20 matches between ' + s3[1] + ' and ' + s4[1] +'.png', img6)

# set that atleast 10 matches are to be there to find the objects
MIN_MATCH_COUNT = 10

# condition set that atleast 10 matches are to be there to find the objects
if len(good_matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    # compute the homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    print('The total number of good matches are: ' + str(mask.size))
    print('The total number of inliers (good matches providing correct estimation and total numbers consistent with the computed homography)) are: ' + str(np.sum(matchesMask)))
    print('The homography matrix when ' + img_detect + ' is matched with ' + img_target + ':')
    print(M)

    # capture the height and width of the image
    h, w = image1.shape

    # If enough matches are found, we extract the locations of matched keypoints in both the images
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # The locations of matched keypoints  are passed to find the perpective transformation
    dst = cv2.perspectiveTransform(pts,M)

    # Once we get this 3x3 transformation matrix, we use it to transform the corners of queryImage to corresponding points in trainImage.
    # Then we draw it.
    image2 = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

# Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed)
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

# draw the final SIFT matching
img3 = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, **draw_params)
cv2.imshow('Final SIFT Match between ' + img_detect + ' and ' + img_target, img3)
cv2.imwrite('HW3_Data/'+'Final SIFT Match between '+ s3[1] + ' and ' + s4[1] +'.png',img3)

# wait for ESC key to exit
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
    plt.close('all')
