# _*_ coding: UTF-8 _*_
# 2020/4/17 20:43
# PyCharm
# Create by:LIUMINXUAN
# E-mail:liuminxuan1024@gmail.com
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def ReadImageAndSave(image_path):
    """
    以IMREAD_GRAYSCALE的方式读入图片，并保存
    :param image_path:图像路径
    :return:返回读取出的图像矩阵
    """
    src_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.namedWindow("src", cv.WINDOW_AUTOSIZE)
    cv.imshow("src", src_image)
    cv.waitKey(2000)
    cv.destroyAllWindows()
    print("Height:%s, Width:%s" % (src_image.shape[0], src_image.shape[1]))
    ret = input("是否另存为到gray文件夹? Y/N")
    if ret.lower() == 'y':
        index = input("请输入文件名:")
        cv.imwrite("./gray/" + str(index) + ".bmp", src_image)
    return src_image


def ReadGrayImage(image_path):
    """
    读取灰度图像
    :param image_path: 图像的相对路径
    :return: 返回读取出的图像矩阵
    """
    src_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.namedWindow("src_image", cv.WINDOW_AUTOSIZE)
    cv.imshow("src_image", src_image)
    cv.waitKey()
    cv.destroyAllWindows()
    print(src_image.shape)
    return src_image


def ReadColorImage(image_path):
    """
    读取RGB图像
    :param image_path: 图像的相对路径
    :return: 返回读取出的图像矩阵
    """
    src_image = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.namedWindow("src_image", cv.WINDOW_AUTOSIZE)
    cv.imshow("src_image", src_image)
    cv.waitKey()
    cv.destroyAllWindows()
    print(src_image.shape)
    return src_image


# (256, 256) 512 * 512
def GrayLevelCount(image):
    """
    统计传入图像的每个灰度级对应的像素点数
    :param image: 传入图像矩阵
    :return: 返回灰度级计数序列
    """
    pix_list = [0] * 256  # 灰度级计数序列 256 灰度级
    for Row in range(image.shape[0]):
        for Column in range(image.shape[1]):
            pix = image[Row, Column]
            pix_list[pix] += 1
    return pix_list


def DrawHistogram(pix_list):
    """
    绘制灰度直方图
    :param pix_list: 传入统计出的灰度级序列
    """
    y = pix_list
    x = [i for i in range(256)]
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Gray level")
    plt.ylabel("Number of pixels")
    plt.plot(x, y)
    plt.show()
    plt.xlim([0, 256])


def DrawHistogram_plot(src_image):
    """
    调用函数绘制直方图
    :param src_image:
    :return:
    """
    plt.hist(src_image.ravel(), 256, [0, 255])
    plt.show()


def ColorImageHistogram(src_image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([src_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def HistogramEqualization_sys(src_image):
    """
    调用系统函数，完成全局的直方图均衡化
    :param src_image:
    :return:
    """
    gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("HistogramEqualization_sys", dst)
    cv.waitKey()
    cv.destroyAllWindows()


def LocalHistogramEqualization(src_image):
    gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("LocalHistogramEqualization", dst)
    cv.waitKey()
    cv.destroyAllWindows()


def Create_RGB_Histogram(src_image):
    height, width, channel = src_image.shape
    RGB_Hist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for Row in range(height):
        for Column in range(width):
            pix_B = src_image[Row, Column, 0]
            pix_G = src_image[Row, Column, 1]
            pix_R = src_image[Row, Column, 2]
            index = np.int(pix_B / bsize) * 16 * 16 + np.int(pix_G / bsize) * 16 + np.int(pix_R / bsize)
            RGB_Hist[np.int(index), 0] += 1
    return RGB_Hist


def HistogramComparison(src_image_1, src_image_2):
    Hist_1 = Create_RGB_Histogram(src_image_1)
    Hist_2 = Create_RGB_Histogram(src_image_2)
    Match_1 = cv.compareHist(Hist_1, Hist_2, cv.HISTCMP_BHATTACHARYYA)
    Match_2 = cv.compareHist(Hist_1, Hist_2, cv.HISTCMP_CORREL)
    Match_3 = cv.compareHist(Hist_1, Hist_2, cv.HISTCMP_CHISQR)
    print("巴氏距离:%s, 相关性:%s, 卡方:%s" % (Match_1, Match_2, Match_3))


def AddNoise(noise_type, image):
    if noise_type == "Gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        Gaussian = np.random.normal(mean, sigma, (row, col, ch))
        Gaussian = Gaussian.reshape(row, col, ch)
        noise = image + Gaussian
        return noise
    elif noise_type == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def ManualHistogramEqualization(src_image):
    pix_list = GrayLevelCount(src_image)
    Normal_pix_list = [0] * 256
    for index in range(len(pix_list)):
        Normal_pix_list[index] = pix_list[index] / src_image.size
    print(Normal_pix_list)
    y = Normal_pix_list
    x = [i for i in range(256)]
    plt.figure()
    plt.title("NormalizedHistogram")
    plt.xlabel("Gray level")
    plt.ylabel("Number of pixels")
    plt.plot(x, y)
    plt.show()
    plt.xlim([0, 256])
    Total_pix_list = [0] * 256
    for i in range(len(Normal_pix_list)):
        for j in range(i):
            Total_pix_list[i] += Normal_pix_list[j]
    print(Total_pix_list)
    y = Total_pix_list
    x = [i for i in range(256)]
    plt.figure()
    plt.title("CumulativeHistogram")
    plt.xlabel("Gray level")
    plt.ylabel("Number of pixels")
    plt.plot(x, y)
    plt.show()
    plt.xlim([0, 256])
    for pix in pix_list:
        print(pix / src_image.size)
    print(src_image.size)
    GrayValAfterTrans = [0] * 256
    for i in range(len(Total_pix_list)):
        GrayValAfterTrans[i] = int((len(Total_pix_list)-1) * Total_pix_list[i] + 0.5)
    # print(GrayValAfterTrans)
    image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
    for Row in range(src_image.shape[0]):
        for Column in range(src_image.shape[1]):
            pix = src_image[Row, Column]
            pix_list[pix] = GrayValAfterTrans[pix]
            image[Row, Column] = pix_list[pix]
    return image


if __name__ == "__main__":
    # img_path = "../space-images/couplegray.bmp"
    gray_image_path = "./gray/couple.bmp"
    # color_image_path = "../space-images/couplegray.bmp"
    src_gray = ReadGrayImage(gray_image_path)
    # src_color = ReadColorImage(color_image_path)
    pix_list = GrayLevelCount(src_gray)
    # DrawHistogram(pix_list)
    # DrawHistogram_plot(src_gray)
    # ColorImageHistogram(src_color)
    # HistogramEqualization_sys(src_color)
    # LocalHistogramEqualization(src_gray)
    # HistogramComparison(src_color, src_color)
    # ret = AddNoise("speckle", src_color)
    # cv.imshow("123", ret)
    # cv.waitKey()
    # cv.destroyAllWindows()
    src = ManualHistogramEqualization(src_gray)
    pix_list = GrayLevelCount(src)
    DrawHistogram(pix_list)
    # src = GrayLevelCount(src_gray)
    # print(src)
    cv.imshow("src", src)
    cv.waitKey()
    cv.destroyAllWindows()
