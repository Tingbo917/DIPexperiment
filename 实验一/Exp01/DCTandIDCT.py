# _*_ coding: UTF-8 _*_
# 2020/4/14 15:56 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
import time
import glob
import os
import cv2 as cv
import numpy as np
import pandas as pd

WSI_MASK_PATH = "./result"
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
paths.sort()

for path in paths:
    print("分块图像处理结果:%s" % path)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    print("====================================")
    img = np.float32(img)
    result_dct = cv.dct(img)
    pd.set_option('display.max_columns', None)  # 显示完整的列
    pd.set_option('display.max_rows', None)  # 显示完整的行
    print("DCT结果:", result_dct)
    print("====================================")
    result_idct = cv.idct(result_dct)
    print("IDCT结果:", result_idct)
    print("====================================")
