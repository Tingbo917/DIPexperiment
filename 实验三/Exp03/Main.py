# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 8:34 下午
# @Author  : Liu minxuan
# @Email   : liuminxuan2016@gmail.com
# @File    : Main.py
import os
import cv2 as cv
import numpy as np
from scipy import signal
from matplotlib import pyplot
smooth_image_path = "images/smooth/danghui_jiaoyan.bmp"  # 需要进行平滑处理的图像路径
sharpen_image_path = "images/sharpen/dayanta_mohu.bmp"  # 需要进行锐化处理的图像路径


# 读入原图
src_smooth_img = cv.imread(smooth_image_path, cv.IMREAD_GRAYSCALE)
src_sharpen_img = cv.imread(sharpen_image_path, cv.IMREAD_GRAYSCALE)
# cv.imshow("src_smooth_img", src_smooth_img)  # 显示原图
cv.imshow("src_sharpen_img", src_sharpen_img)

# 调用函数实现低通滤波
# 低通均值滤波的另一种处理方法
# # TheMeanSmooth.py
Low_pass_filter = cv.blur(src_smooth_img, (3, 3))  # 参数kernel是奇数，可自由调整
# cv.imshow("Low pass filter", Low_pass_filter)  # 显示处理后的图像


# 调用函数实现中值滤波
# 中值滤波的另一种处理方法
# TheMedianSmooth.py
Median_filter = cv.medianBlur(src_smooth_img, 3)
# cv.imshow("Median filter", Median_filter)


# 调用函数实现高斯滤波，此高斯滤波与高斯有所区别
# 高斯滤波的另一种处理方法
# # TheGaussianSmooth.py
Gaussian_filter = cv.GaussianBlur(src_smooth_img, (3, 3), 0)
# cv.imshow("Gaussian filter", Gaussian_filter)


# 低通卷积掩膜: Box 和 高斯掩膜
mask_Box = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
mask_Gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
print(mask_Box)
print(mask_Gaussian)
mask_Box_res = cv.filter2D(src_smooth_img, -1, mask_Box)
mask_Gaussian_res = cv.filter2D(src_smooth_img, -1, mask_Gaussian)
cv.imshow("src_smooth_img", src_smooth_img)
cv.imshow("mask_Box_res", mask_Box_res)
cv.imshow("mask_Gaussian_res", mask_Gaussian_res)

# 拉普拉斯掩膜图像锐化
kernel_sharpen_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 无对角线方向的拉普拉斯掩膜
kernel_sharpen_2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 含对角线方向的拉普拉斯掩膜
kernel_LoG = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
output_1 = cv.filter2D(src_sharpen_img, -1, kernel_sharpen_1)
output_2 = cv.filter2D(src_sharpen_img, -1, kernel_sharpen_2)
output_3 = cv.filter2D(src_sharpen_img, -1, kernel_LoG)
cv.imshow("sharpen_1 Image", output_1)
cv.imshow("sharpen_2 Image", output_2)
cv.imshow("kernel_LoG", output_3)


# 实现手动输入掩膜，而后进行卷积运算
kernel = []  # 空的list，用于存储输入的数据
count = 0  # 计数，标识输入到第几个参数
sums = 0  # 求和，计算输入参数的数值总和，便于归一化处理
param_order = input("请输入掩膜的阶数(奇数):")  # 从控制台接收输入，确定掩膜的阶数
# 循环输入阶段
param_order = int(param_order)
for rows in range(param_order):
    kernel.append([])
    for cols in range(param_order):
        count += 1
        val = int(input("请输入第%d整数:\n" % count))
        kernel[rows].append(val)
        sums += val
Mask = np.array(kernel)
print(Mask)
print(Mask/sums)
Mask = Mask / sums  # 归一化处理后的掩膜


cv.waitKey(0)
cv.destroyAllWindows()



