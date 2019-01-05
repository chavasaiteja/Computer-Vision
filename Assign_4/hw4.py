import cv2
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image1",required = True, type = str, help = "Path to image 1")
ap.add_argument("-j","--image2",required = True, type = str, help = "Path to image 2")

args = vars(ap.parse_args())

image_1 = args["image1"]
image_2 = args["image2"]

MIN_MATCH_COUNT = 10

#TODO: Load Different Image Pairs
img1=cv2.imread(image_1)
img2=cv2.imread(image_2)

#Gray images 
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#TODO: Replace K with given Intrinsic Matrix
K = np.array([[518.86, 0.0, 285.58],
              [0.0, 519.47, 213.74],
              [0.0,   0.0,   1.0]])

###############################
#1----SIFT feature matching---#
###############################

#detect sift features for both images
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print('The features found in image1 : ' + str(des1.shape[0]))
print('The features found in image2 are :' + str(des2.shape[0]))

x = image_1.split(".")[0]
y = image_2.split(".")[0]

#Drawing keypoints on the images
img4 = cv2.drawKeypoints(gray1,kp1,gray1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv2.drawKeypoints(gray2,kp2,gray2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Displaying the key point descriptors
cv2.imshow('Image1 with SIFT keypoints',img4)
cv2.imwrite(x+"SIFT keypoints.png",img4)
cv2.waitKey(0)
cv2.imshow('Image2 with SIFT keypoints',img5)
cv2.imwrite(y+"SIFT keypoints.png",img5)
cv2.waitKey(0)

#use flann to perform feature matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print("No of matches before RANSAC :"+str(len(good)))
if len(good)>MIN_MATCH_COUNT:
    p1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    p2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

img_siftmatch = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow('Matches between ' + image_1 + ' and ' + image_2, img_siftmatch)
cv2.waitKey(0)
#s = x+"Macthes between "+ x +" and "+ y +".png"
#cv2.imwrite(s,img_siftmatch)
#cv2.waitKey(0)
#cv2.imwrite('../results/sift_match.png',img_siftmatch)

#########################
#2----essential matrix--#
#########################
E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0);

matchesMask = mask.ravel().tolist()	

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

print("No of matches after RANSAC :"+str(matchesMask.count(1)))
img_inliermatch = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow("Inlier matches are",img_inliermatch)
cv2.waitKey(0)
#s = x+"inlier_match between "+ x +" and "+ y +".png"
#cv2.imwrite(s,img_inliermatch)
print("Essential matrix is :") 
print(E)

# Verify the rank is 2
print("Essential Matrix rank is :"+str(matrix_rank(E)))

####################
#3----recoverpose--#
####################

points, R, t, mask = cv2.recoverPose(E, p1, p2)
print("Rotation Matrix is :")
print(R)
print("Translation Matrix is :")
print(t)
# p1_tmp = np.expand_dims(np.squeeze(p1), 0)
p1_tmp = np.ones([3, p1.shape[0]])
p1_tmp[:2,:] = np.squeeze(p1).T
p2_tmp = np.ones([3, p2.shape[0]])
p2_tmp[:2,:] = np.squeeze(p2).T
#print((np.dot(R, p2_tmp) + t) - p1_tmp)

#######################
#4----triangulation---#
#######################

#calculate projection matrix for both camera
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(K,  M_l)
P_r = np.dot(K,  M_r)

#Camera Matrixes
print("Camrera Matrix for the first camera is :")
print(P_l)
print("Camrera Matrix for the second camera is :")
print(P_r)

# undistort points
p1 = p1[np.asarray(matchesMask)==1,:,:]
p2 = p2[np.asarray(matchesMask)==1,:,:]
p1_un = cv2.undistortPoints(p1,K,None)
p2_un = cv2.undistortPoints(p2,K,None)
p1_un = np.squeeze(p1_un)
p2_un = np.squeeze(p2_un)

#triangulate points this requires points in normalized coordinate
point_4d_hom = cv2.triangulatePoints(M_l, M_r, p1_un.T, p2_un.T)
point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_try = point_3d
point_3d = point_3d[:3, :].T

points_2d_perspective = np.dot(M_l,point_try)
point_2d = points_2d_perspective/np.tile(points_2d_perspective[-1:],(3,1))
point_2d = point_2d[:2,:].T

diff1 = np.linalg.norm(point_2d-p1_un)
diff2 = np.linalg.norm(point_2d-p2_un)
print("Avg euclidean distance or error w.r.t camera 1 :",diff1/point_2d.shape[0])
#print("Avg euclidean distance or error w.r.t camera 2 :",diff2/point_2d.shape[0])
#############################
#5----output 3D pointcloud--#
#############################
#TODO: Display 3D points
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(point_3d[:,0],point_3d[:,1],point_3d[:,2])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
