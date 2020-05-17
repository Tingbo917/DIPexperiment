import cv2
import numpy as np
from scipy import signal


def roberts(I, _boundary='fill', _fillvalue=0):
    # 图像的高，宽
    H1, W1 = I.shape[0:2]

    # 卷积核的尺寸
    H2, W2 = 2, 2

    # 卷积核1 和 锚点的位置
    R1 = np.array([[1, 0], [0, -1]], np.float32)
    kr1, kc1 = 0, 0

    # 计算full卷积
    IconR1 = signal.convolve2d(I, R1, mode='full', boundary=_boundary, fillvalue=_fillvalue)
    IconR1 = IconR1[H2 - kr1 - 1:H1 + H2 - kr1 - 1, W2 - kc1 - 1:W1 + W2 - kc1 - 1]

    # 卷积核2 和 锚点的位置
    R2 = np.array([[0, 1], [-1, 0]], np.float32)
    kr2, kc2 = 0, 1
    # 再计算full卷积
    IconR2 = signal.convolve2d(I, R2, mode='full', boundary=_boundary, fillvalue=_fillvalue)
    IconR2 = IconR2[H2 - kr2 - 1:H1 + H2 - kr2 - 1, W2 - kc2 - 1:W1 + W2 - kc2 - 1]

    return (IconR1, IconR2)


if __name__ == '__main__':
    I = cv2.imread('../images/danghui_yuantu.bmp', cv2.IMREAD_GRAYSCALE)
    # 显示原图
    cv2.imshow('origin', I)
    # 卷积，注意边界一般扩充采用的symm
    IconR1, IconR2 = roberts(I, 'symm')

    # 45度方向上的边缘强度的灰度级显示
    IconR1 = np.abs(IconR1)
    edge45 = IconR1.astype(np.uint8)
    cv2.imshow('edge45', edge45)

    # 135度方向上的边缘强度的灰度级显示
    IconR2 = np.abs(IconR2)
    edge135 = IconR2.astype(np.uint8)
    cv2.imshow('edge135', edge135)

    # 用平方和的开方来衡量最后输出的边缘
    edge = np.sqrt(np.power(IconR1, 2.0) + np.power(IconR2, 2.0))
    edge = np.round(edge)
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)

    # 显示边缘
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
