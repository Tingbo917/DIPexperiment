import cv2 as cv
import numpy as np
from numpy.core._multiarray_umath import ndarray


def HorizontalAndVerticalGradientOperator(image):
    image = cv.resize(image, (600, 600))
    new_image = np.copy(image)
    rows, cols = image.shape
    kernel_Level = np.array([[-1, 1], [0, 0]])
    kernel_Vertical = np.array([[-1, 0], [1, 0]])
    for x in range(rows):
        for y in range(cols):
            block = image[x:x + 2, y:y + 2]
            dx = cv.filter2D(block, -1, kernel_Level)
            sum_x = abs(dx.sum())
            dy = cv.filter2D(block, -1, kernel_Vertical)
            sum_y = abs(dy.sum())
            var_xy = sum_x + sum_y
            new_image[x, y] = var_xy
    return new_image


def HorizontalAndVerticalGradientOperatorWithThreshold(image, mode='1', minVar=64, maxVar=128, Threshold=100):
    # image = cv.resize(image, (720, 720))
    new_image: ndarray = np.copy(image)
    rows, cols = image.shape
    kernel_Level = np.array([[-1, 1], [0, 0]])
    kernel_Vertical = np.array([[-1, 0], [1, 0]])
    if mode == '1':  # 不做阈值处理
        for x in range(rows):
            for y in range(cols):
                block = image[x:x + 2, y:y + 2]
                # dx = cv.filter2D(block, -1, kernel_Level)
                dx = block*kernel_Level
                sum_x = abs(dx.sum())
                # dy = cv.filter2D(block, -1, kernel_Vertical)
                dy = block*kernel_Vertical
                sum_y = abs(dy.sum())
                var_xy = sum_x + sum_y
                new_image[x, y] = var_xy
        return new_image
    elif mode == '2':  # 当G(x,y)>T时，取G(x,y)的值,否则取f(x,y)
        for x in range(rows):
            for y in range(cols):
                block = image[x:x + 2, y:y + 2]
                # dx = cv.filter2D(block, -1, kernel_Level)
                dx = block * kernel_Level
                sum_x = abs(dx.sum())
                # dy = cv.filter2D(block, -1, kernel_Vertical)
                dy = block * kernel_Vertical
                sum_y = abs(dy.sum())
                var_xy = sum_x + sum_y
            if var_xy > Threshold:
                new_image[x, y] = var_xy
            else:
                new_image[x, y] = image[x, y]
        return new_image
    elif mode == '3':  # 当G(x,y)>T时，取maxVar,否则取f(x,y)
        for x in range(rows):
            for y in range(cols):
                block = image[x:x + 2, y:y + 2]
                # dx = cv.filter2D(block, -1, kernel_Level)
                dx = block * kernel_Level
                sum_x = abs(dx.sum())
                # dy = cv.filter2D(block, -1, kernel_Vertical)
                dy = block * kernel_Vertical
                sum_y = abs(dy.sum())
                var_xy = sum_x + sum_y
                if var_xy > Threshold:
                    new_image[x, y] = maxVar
                else:
                    new_image[x, y] = image[x, y]
        return new_image
    elif mode == '4':  # 当G(x,y)>T时，取G(x,y)的值,否则取minVar
        for x in range(rows):
            for y in range(cols):
                block = image[x:x + 2, y:y + 2]
                # dx = cv.filter2D(block, -1, kernel_Level)
                dx = block * kernel_Level
                sum_x = abs(dx.sum())
                # dy = cv.filter2D(block, -1, kernel_Vertical)
                dy = block * kernel_Vertical
                sum_y = abs(dy.sum())
                var_xy = sum_x + sum_y
                if var_xy > Threshold:
                    new_image[x, y] = var_xy
                else:
                    new_image[x, y] = minVar
        return new_image
    elif mode == '5':  # 当G(x,y)>T时，取maxVar的值,否则取minVar
        for x in range(rows):
            for y in range(cols):
                block = image[x:x + 2, y:y + 2]
                # dx = cv.filter2D(block, -1, kernel_Level)
                dx = block * kernel_Level
                sum_x = abs(dx.sum())
                # dy = cv.filter2D(block, -1, kernel_Vertical)
                dy = block * kernel_Vertical
                sum_y = abs(dy.sum())
                var_xy = sum_x + sum_y
                if var_xy > Threshold:
                    new_image[x, y] = maxVar
                else:
                    new_image[x, y] = minVar
        return new_image
    else:
        print("mode error!")
        return


src = cv.imread("./images/dayanta_yuantu.bmp", cv.IMREAD_GRAYSCALE)
dst_1 = HorizontalAndVerticalGradientOperatorWithThreshold(src, mode='1', Threshold=10)
# dst_2 = HorizontalAndVerticalGradientOperator(src)
cv.imshow("src", src)
cv.imshow("dst", dst_1)

cv.waitKey(0)
cv.destroyAllWindows()
