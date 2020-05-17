# _*_ coding: UTF-8 _*_
# 2020/5/5 3:02 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
import cv2 as cv
import numpy as np
import Sort


# 排序算法
# 各种排序算法在Sort.py
# 线形窗口中值滤波
def LinearWindowFilter(src_img, winSize):
    if winSize % 2 == 0 or winSize == 1:
        print('winSize Error! such as:3, 5, 7, 9....')
        return None
    paddingSize = winSize//2
    height, width = src_img.shape
    img_base = np.zeros((height, width + paddingSize * 2), np.uint8)
    for i in range(height):
        for j in range(paddingSize+1, width):
            img_base[i, j] = src_img[i, j - paddingSize]
    imgOut = np.zeros((height, width), dtype=src_img.dtype)
    for x in range(height):
        for y in range(width):
            line = img_base[x, y:y + winSize].flatten()
            line = Sort.insert_sort(line)
            imgOut[x, y] = line[winSize // 2]
    return imgOut


# 十字形窗口中值滤波
def CrossWindowMedianFilter(src_img, winSize):
    if winSize % 2 == 0 or winSize == 1:
        print('winSize Error! such as:3, 5, 7, 9....')
        return None
    elif winSize == 3:
        CrossWindows = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])
        print(CrossWindows)
    elif winSize == 5:
        CrossWindows = np.array([[0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [1, 1, 1, 1, 1],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0]])
        print(CrossWindows)
    elif winSize == 7:
        CrossWindows = np.array([[0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0]])
        print(CrossWindows)
    elif winSize == 9:
        CrossWindows = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0]])
        print(CrossWindows)
    paddingSize = winSize // 2
    height, width = src_img.shape
    matBase = np.zeros((height + paddingSize * 2, width + paddingSize * 2), dtype=src_img.dtype)
    matBase[paddingSize:-paddingSize, paddingSize:-paddingSize] = src_img
    for r in range(paddingSize):
        matBase[r, paddingSize:-paddingSize] = src_img[0, :]
        matBase[-(1 + r), paddingSize:-paddingSize] = src_img[-1, :]
        matBase[paddingSize:-paddingSize, r] = src_img[:, 0]
        matBase[paddingSize:-paddingSize, -(1 + r)] = src_img[:, -1]
    matOut = np.zeros((height, width), dtype=src_img.dtype)
    for x in range(height):
        for y in range(width):
            line = matBase[x:x + winSize, y:y + winSize]
            ret = Sort.MatrixMultiplication(line, CrossWindows)
            lst = Sort.RemoveZero(ret)
            lst = sorted(lst)
            if lst is None:
                matOut[x, y] = src_img[x, y]
            elif len(lst) == 1:
                matOut[x, y] = lst[0]
            elif len(lst) == 2:
                matOut[x, y] = lst[1]
            elif len(lst) == 3:
                matOut[x, y] = lst[1]
            elif len(lst) == 4:
                matOut[x, y] = lst[1]
            elif len(lst) == 5:
                matOut[x, y] = lst[2]
            elif len(lst) == 6:
                matOut[x, y] = lst[2]
            elif len(lst) == 7:
                matOut[x, y] = lst[3]
            elif len(lst) == 8:
                matOut[x, y] = lst[3]
            elif len(lst) == 9:
                matOut[x, y] = lst[4]
    return matOut


# 方形窗口中值滤波
def SquareWindowMedianFilter_1(src_img, winSize=3):
    if winSize % 2 == 0 or winSize == 1:
        print('winSize Error! such as:3, 5, 7, 9....')
        return None
    paddingSize = winSize//2
    height, width = src_img.shape
    matBase = np.zeros((height + paddingSize * 2, width + paddingSize * 2), dtype=src_img.dtype)
    matBase[paddingSize:-paddingSize, paddingSize:-paddingSize] = src_img
    for r in range(paddingSize):
        matBase[r, paddingSize:-paddingSize] = src_img[0, :]
        matBase[-(1 + r), paddingSize:-paddingSize] = src_img[-1, :]
        matBase[paddingSize:-paddingSize, r] = src_img[:, 0]
        matBase[paddingSize:-paddingSize, -(1 + r)] = src_img[:, -1]
    matOut = np.zeros((height, width), dtype=src_img.dtype)
    for x in range(height):
        for y in range(width):
            line = matBase[x:x + winSize, y:y + winSize].flatten()
            line = np.sort(line)
            matOut[x, y] = line[(winSize * winSize) // 2]
    return matOut


def SquareWindowMedianFilter_2(src_img, winSize):
    rows, cols = src_img.shape
    winH, winW = winSize
    halfWinH = (winH - 1) // 2
    halfWinW = (winW - 1) // 2
    medianBlurImage = np.zeros(src_img.shape, src_img.dtype)
    for r in range(rows):
        for c in range(cols):
            if r - halfWinH < 0:
                rTop = 0
            else:
                rTop = r - halfWinH
            if r + halfWinH > rows - 1:
                rBottom = rows - 1
            else:
                rBottom = r + halfWinH
            if c - halfWinW < 0:
                cLeft = 0
            else:
                cLeft = c - halfWinW
            if c + halfWinW > cols - 1:
                cRight = cols - 1
            else:
                cRight = c + halfWinW
            region = src_img[rTop:rBottom + 1, cLeft:cRight + 1]
            medianBlurImage[r][c] = np.median(region)
    medianBlurImage = medianBlurImage.astype(np.uint8)
    return medianBlurImage


def SquareWindowMedianFilter_3(src_img, winSize):
    img = np.array(src_img)
    line = []
    for i in range(int(winSize / 2), img.shape[0] - int(winSize / 2)):
        for j in range(int(winSize / 2), img.shape[1] - int(winSize / 2)):
            for a in range(-int(winSize / 2), -int(winSize / 2) + winSize):
                for b in range(-int(winSize / 2), -int(winSize / 2) + winSize):
                    line.append(img[i + a, j + b])
            line.sort()
            img[i, j] = line[int(winSize * winSize / 2)]
            line = []
    return img


# 调用OpenCV函数中值滤波
def CallOpencvFunctionMedianFilter(src_img, winSize):
    img = cv.medianBlur(src_img, winSize)
    return img


if __name__ == "__main__":
    src_image = cv.imread("./img/jiaoyan/danghui_jiaoyan.bmp")
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    cv.imshow("src image", src_image)
    # ret_1 = SquareWindowMedianFilter_1(src_image, 3)
    # ret_2 = SquareWindowMedianFilter_2(src_image, (3, 3))
    # ret_3 = SquareWindowMedianFilter_3(src_image, 3)
    # ret_4 = CallOpencvFunctionMedianFilter(src_image, 3)
    # ret_5 = LinearWindowFilter(src_image, 9)
    # cv.imshow("ret_1", ret_1)
    # cv.imshow("ret_2", ret_2)
    # cv.imshow("ret_3", ret_3)
    # cv.imshow("ret_4", ret_4)
    # cv.imshow("ret_5", ret_5)
    # test = np.array([[7, 6, 8, 7],
    #                  [6, 3, 4, 15],
    #                  [1, 1, 1, 1],
    #                  [4, 3, 1, 2]])
    # CrossWindowMedianFilter(test, 3)
    ret_6 = CrossWindowMedianFilter(src_image, 3)
    cv.imshow("ret_6", ret_6)
    cv.waitKey(8000)
    cv.destroyAllWindows()
