import cv2
import numpy as np

img = cv2.imread('6_3_s.bmp')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)

edge_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
edge_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)    
edge = np.sqrt(edge_x**2 + edge_y**2)    # image can be normalized to 
                                         # fit into 0..255 color space
cv2.imwrite('openCvSobel.jpg', edge)