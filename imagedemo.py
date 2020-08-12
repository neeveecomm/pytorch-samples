import cv2

img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('test.png', img)
cv2.imshow('ImageDemo', img)
cv2.waitKey()

cv2.destroyAllWindows()