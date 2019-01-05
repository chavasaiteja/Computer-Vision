import cv2

image = cv2.imread("federer.jpg")
print(type(image))

r = image.shape[1]*1.1 / image.shape[1]
dim = (int(image.shape[1]*1.1), int(image.shape[0] * r))
 
# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)

print(type(resized)) 
