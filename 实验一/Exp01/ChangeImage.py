# _*_ coding: UTF-8 _*_
# 2020/4/13 20:20 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
import cv2 as cv
img_path = "../images/LENA.BMP"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
cv.imshow("lena", img)
cv.waitKey(2000)
cv.destroyAllWindows()
cv.imwrite("../Exp01/gray/lena.png", img)
print(img.shape)
