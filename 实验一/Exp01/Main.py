# _*_ coding: UTF-8 _*_
# 2020/4/14 11:38 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
from PIL import Image
import numpy as np
import cv2 as cv
import math
import sys
import pandas as pd


# 读取图像:（1）读取RGB图像后，转换为灰度图像保存（2）在读取图像的同时完成转换，不保存
def ReadImageWithColor(src_image):
    """
    以RGB的方式读取图像，此时图像有三个通道，并显示，选择是否保存
    :param src_image: 输入参数为图片的路径
    :return:返回读取出的矩阵
    """
    src_image = cv.imread(src_image, cv.IMREAD_COLOR)  # 使用OpenCV内置函数以RGB方式读入
    cv.namedWindow("Color", cv.WINDOW_AUTOSIZE)
    cv.imshow("Color", src_image)  # 显示图像
    cv.waitKey(2000)  # 图形界面停留4s
    cv.destroyAllWindows()
    ret = input("是否另存为到color文件夹? Y/N")
    if ret.lower() == 'y':
        index = input("请输入文件名:")
        cv.imwrite("./color/" + str(index) + ".png", src_image)
    print("src_image的类型:", type(src_image))
    print("图像的高度为:%s像素, 图像的宽度为:%s像素,通道数为:%s" % (src_image.shape[0], src_image.shape[1], src_image.shape[2]))
    print("图像有%s个像素点" % src_image.size)
    return src_image


def ReadImageWithGray(src_image):
    """
    以灰度的方式读入图像，此时图像只有单通道
    :param src_image: 传入的参数为图像的完整路径
    :return: 返回处理过的图像矩阵
    """
    src_image = cv.imread(src_image, cv.IMREAD_GRAYSCALE)
    cv.namedWindow("Gray", cv.WINDOW_AUTOSIZE)  # 给窗口起名字 根据图像大小自动匹配窗口大小
    cv.imshow("Gray", src_image)  # 显示图像
    cv.waitKey(0)
    cv.destroyAllWindows()  # 销毁窗口
    ret = input("是否另存为到gray文件夹? Y/N")
    if ret.lower() == 'y':
        index = input("请输入文件名:")
        cv.imwrite("./gray/" + str(index) + ".png", src_image)
    print("src_image的类型:", type(src_image))
    print("图像的高度为:%s像素, 图像的宽度为:%s像素" % (src_image.shape[0], src_image.shape[1]))
    print("图像有%s个像素点" % src_image.size)
    return src_image


# 计算图像的灰度平均值
def CalGrayAverageValueOfImage(src_image):
    """
    计算图像的灰度平均值，先将彩色图像转换为灰度图像 可用opencv的函数 cv.mean()求均值
    :param src_image: 传入的参数为Read函数的返回值
    :return: 返回求得的灰度平均值
    """
    SumOfPixGrayValue = 0 # 求像素点的灰度值之和
    for Row in range(src_image.shape[0]):
        for Column in range(src_image.shape[1]):
            px = src_image[Row, Column]
            SumOfPixGrayValue += px
    print("图像灰度值总和为:%s" % SumOfPixGrayValue)
    print("平均灰度值为:%s" % (SumOfPixGrayValue / src_image.size))
    print("调用opencv函数计算的灰度值为:%s", cv.mean(src_image))
    AverageValue = SumOfPixGrayValue / src_image.size
    return AverageValue


# 计算图像的方差
def CalVarianceOfImage(src_image):
    """
    计算图像的方差 也可调用opencv函数
    :param src_image:
    :return:
    """
    AverageVal = CalGrayAverageValueOfImage(src_image)
    TotalPixels = src_image.shape[0] * src_image.shape[1]
    SumOfSquares = 0
    for Row in range(src_image.shape[0]):
        for Column in range(src_image.shape[1]):
            px = src_image[Row, Column]
            SumOfSquares += (px - AverageVal) * (px - AverageVal)
    Variance = SumOfSquares / TotalPixels
    print("图像的方差为:%s" % Variance)
    mean_img, stddev_img = cv.meanStdDev(src_image)
    print("调用opencv函数计算图像的方差为:%s" % np.var(src_image))
    print("调用opencv函数计算的的均值为:%s,标准差为:%s" % (mean_img, stddev_img))
    return Variance


# 计算图像的标准差
def CalGrayStandardDeviationOfImage(src_image):
    """
    计算图像的灰度标准差
    :param src_image:
    :return:
    """
    Var = CalVarianceOfImage(src_image)
    GrayStdDeviation = math.sqrt(Var)
    print("图像的灰度标准差为:%s" % GrayStdDeviation)
    return GrayStdDeviation


# 计算图像的协方差值
def CalCovarianceMatrix(src_image_1, src_image_2):
    """

    :param src_image_1:
    :param src_image_2:
    :return:
    """
    covxy = np.cov(src_image_1, src_image_2)
    print(covxy)


# 计算图像的相关系数
def CalImageCorrelationCoefficient(src_image_1, src_image_2):
    coefxy = np.corrcoef(src_image_1, src_image_2)
    print(coefxy)


# 分割图像为4X416块，保存在result文件夹下
# SplitImage.py文件

if __name__ == "__main__":
    # src_img_1 = ReadImageWithGray("../images/WindowsLogo.jpg")
    # src_img_2 = ReadImageWithGray("../images/LinuxLogo.jpg")
    src = ReadImageWithGray("../images/lena.jpg")
    # CalVarianceOfImage(src)
    # CalGrayAverageValueOfImage(src)
    # CalGrayStandardDeviationOfImage(src)
    # CalCovarianceMatrix(src_img_1, src_img_2)
    # CalImageCorrelationCoefficient(src_img_1, src_img_2)

