# _*_ coding: UTF-8 _*_
# 2020/5/5 15:41 
# PyCharm  
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
# Python实现各种排序算法
import numpy as np


# 1.插入排序
def insert_sort(lst):
    for i in range(1, len(lst)):
        x = lst[i]
        j = i
        while j > 0 and lst[j - 1] > x:
            lst[j] = lst[j - 1]
            j -= 1
        lst[j] = x
    return lst


# 2.选择排序
def select_sort(lst):
    for i in range(len(lst) - 1):
        k = i
        for j in range(len(lst)):
            if lst[j] < lst[k]:
                k = j
        if i != k:
            lst[i], lst[k] = lst[k], lst[i]
    return lst


# 3.冒泡排序
def bubble_sort_1(lst):
    for i in range(len(lst)):
        for j in range(1, len(lst) - i):
            if lst[j - 1] > lst[j]:
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
    return lst


def bubble_sort_2(lst):
    for i in range(len(lst)):
        found = False
        for j in range(1, len(lst) - i):
            if lst[j - 1] > lst[j]:
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
                found = True
        if not found:
            break
    return lst


# 4.快速排序
def quick_sort(lst, left, right):
    if left >= right:
        return
    i = left
    j = right
    pivot = lst[i]
    while i < j:
        while i < j and lst[j] >= pivot:
            j -= 1
        if i < j:
            lst[i] = lst[j]
            i += 1
        while i < j and lst[i] <= pivot:
            i += 1
        if i < j:
            lst[j] = lst[i]
            j -= 1
        lst[i] = pivot
        quick_sort(lst, left, i - 1)
        quick_sort(lst, i + 1, right)
    return lst


# 矩阵对应元素相乘的函数
def MatrixMultiplication(matrix_1, matrix_2):
    height, width = matrix_1.shape
    retMatrix = np.zeros((height, width), np.uint8)
    for rows in range(height):
        for cols in range(width):
            retMatrix[rows, cols] = matrix_1[rows, cols] * matrix_2[rows, cols]
    return retMatrix


# 移除0
def RemoveZero(matrix):
    height, width = matrix.shape
    var = []
    for rows in range(height):
        for cols in range(width):
            if matrix[rows, cols] != 0:
                var.append(matrix[rows, cols])
            else:
                continue
    if var is None:
        var = [0, 0, 0]
    return var
