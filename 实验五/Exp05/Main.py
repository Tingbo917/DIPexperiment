import cv2 as cv
import numpy as np


# Sobel算子
def Call_OpenCV_Function_Sobel(src_img):
    dx = cv.Sobel(src_img, cv.CV_64F, 1, 0)
    dx = cv.convertScaleAbs(dx)
    dy = cv.Sobel(src_img, cv.CV_64F, 0, 1)
    dy = cv.convertScaleAbs(dy)
    dst = cv.addWeighted(dx, 0.5, dy, 0.5, 0)
    return dst


def SobelOperator(roi, operator_type):
    if operator_type == "horizontal":
        sobel_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == "vertical":
        sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        raise Exception("type Error")
    result = np.abs(np.sum(roi * sobel_operator))
    return result


def SobelAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = SobelOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


# Scharr算子
def Call_OpenCV_Function_Scharr(src_img):
    dx = cv.Scharr(src_img, cv.CV_64F, 1, 0)
    dx = cv.convertScaleAbs(dx)
    dy = cv.Scharr(src_img, cv.CV_64F, 0, 1)
    dy = cv.convertScaleAbs(dy)
    dst = cv.addWeighted(dx, 0.5, dy, 0.5, 0)
    return dst


# Laplace算子（二阶）
def Call_OpenCV_Function_Laplace(src_img):
    dst = cv.Laplacian(src_img, cv.CV_64F)
    dst = cv.convertScaleAbs(dst)
    return dst


def LaplaceOperator(roi, operator_type):
    if operator_type == "fourfields":
        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif operator_type == "eightfields":
        laplace_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    else:
        raise Exception("type Error")
    result = np.abs(np.sum(roi * laplace_operator))
    return result


def LaplaceAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = LaplaceOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


# Canny算子
def Call_OpenCV_Function_Canny(src_img, minVal, maxVal):
    dst = cv.Canny(src_img, minVal, maxVal)
    return dst


# Robert算子
def robert_suanzi(src_img):
    r, c = src_img.shape
    robert_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = src_img[x:x + 2, y:y + 2]
                list_robert = robert_sunnzi * imgChild
                src_img[x, y] = abs(list_robert.sum())
    return src_img


def RobertsOperator(roi):
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))


def RobertsAlogrithm(image):
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:image.shape[0], 1:image.shape[1]]


# Prewitt算子
def PreWittOperator(roi, operator_type):
    if operator_type == "horizontal":
        prewitt_operator = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif operator_type == "vertical":
        prewitt_operator = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    else:
        raise Exception("type Error")
    result = np.abs(np.sum(roi * prewitt_operator))
    return result


def PreWittAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = PreWittOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


if __name__ == "__main__":
    # 读取图像
    src_img = cv.imread("./images/danghui_yuantu.bmp", cv.IMREAD_GRAYSCALE)
    # 调用OpenCV的Sobel的函数
    dst_sobel = Call_OpenCV_Function_Sobel(src_img)
    # 调用OpenCV的Scharr的函数
    dst_scharr = Call_OpenCV_Function_Scharr(src_img)
    # 调用OPenCV的Laplacian函数
    dst_laplace = Call_OpenCV_Function_Laplace(src_img)
    # 调用OpenCV的Canny函数
    dst_canny = Call_OpenCV_Function_Canny(src_img, 50, 200)
    # Robert算子
    dst_robert = RobertsAlogrithm(src_img)
    # Sobel算子
    dst_sobel_1 = SobelAlogrithm(src_img, "horizontal")
    # PreWitt算子
    dst_prewitt = PreWittAlogrithm(src_img, "horizontal")
    # Laplace算子
    dst_laplace_1 = LaplaceAlogrithm(src_img, "eightfields")

    # 显示处理结果
    # cv.imshow("src img", src_img)
    # cv.imshow("opencv funcion sobel", dst_sobel)
    # cv.imshow("opencv funcion scharr", dst_scharr)
    # cv.imshow("opencv funcion laplace", dst_laplace)
    # cv.imshow("opencv funcion canny", dst_canny)
    # cv.imshow("robert", dst_robert)
    # cv.imshow("sobel", dst_sobel_1)
    # cv.imshow("prewitt", dst_prewitt)
    # cv.imshow("laplace", dst_laplace_1)
    cv.waitKey(0)
    cv.destroyAllWindows()
