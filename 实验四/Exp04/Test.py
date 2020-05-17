# _*_ coding: UTF-8 _*_
# 2020/5/6 9:10 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
import cv2 as cv
import numpy as np
import Sort
test = np.array([[6, 8, 7],
                 [3, 4, 15],
                 [1, 1, 1]])
CrossWindows = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
ret = Sort.MatrixMultiplication(test, CrossWindows)
print(test)
print(CrossWindows)
print(ret)
ret_1 = Sort.RemoveZero(ret)
print(ret_1)