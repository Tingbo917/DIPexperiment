import cv2 as cv
import numpy as np
import Main
from matplotlib import pyplot as plt

saber = cv.imread("saber.png")
saber = cv.cvtColor(saber, cv.COLOR_BGR2RGB)
plt.imshow(saber)
plt.axis("off")
plt.show()

gray_saber = cv.cvtColor(saber, cv.COLOR_RGB2GRAY)
gray_saber = cv.resize(gray_saber, (200, 200))


def GaussianOperator(roi):
    GaussianKernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    result = np.sum(roi * GaussianKernel / 16)
    return result


def GaussianSmooth(image):
    new_image = np.zeros(image.shape)
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = GaussianOperator(image[i - 1:i + 2, j - 1:j + 2])
    return new_image.astype(np.uint8)


smooth_saber = GaussianSmooth(gray_saber)
plt.subplot(121)
plt.title("Origin Image")
plt.axis("off")
plt.imshow(gray_saber, cmap="gray")
plt.subplot(122)
plt.title("GaussianSmooth Image")
plt.axis("off")
plt.imshow(smooth_saber, cmap="gray")
plt.show()

Gx = Main.SobelAlogrithm(smooth_saber, "horizontal")
Gy = Main.SobelAlogrithm(smooth_saber, "vertical")
G = np.sqrt(np.square(Gx.astype(np.float64)) + np.square(Gy.astype(np.float64)))
cita = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))

plt.imshow(G.astype(np.uint8), cmap="gray")
plt.axis("off")
plt.show()


def NonmaximumSuppression(image, cita):
    keep = np.zeros(cita.shape)
    cita = np.abs(cv.copyMakeBorder(cita, 1, 1, 1, 1, cv2.BORDER_DEFAULT))
    for i in range(1, cita.shape[0] - 1):
        for j in range(1, cita.shape[1] - 1):
            if cita[i][j] > cita[i - 1][j] and cita[i][j] > cita[i + 1][j]:
                keep[i - 1][j - 1] = 1
            elif cita[i][j] > cita[i][j + 1] and cita[i][j] > cita[i][j - 1]:
                keep[i - 1][j - 1] = 1
            elif cita[i][j] > cita[i + 1][j + 1] and cita[i][j] > cita[i - 1][j - 1]:
                keep[i - 1][j - 1] = 1
            elif cita[i][j] > cita[i - 1][j + 1] and cita[i][j] > cita[i + 1][j - 1]:
                keep[i - 1][j - 1] = 1
            else:
                keep[i - 1][j - 1] = 0
    return keep * image


nms_image = NonmaximumSuppression(G, cita)
nms_image = (nms_image * (255 / np.max(nms_image))).astype(np.uint8)
plt.imshow(nms_image, cmap="gray")
plt.axis("off")
plt.show()

MAXThreshold = np.max(nms_image) / 4 * 3
MINThreshold = np.max(nms_image) / 4
usemap = np.zeros(nms_image.shape)
high_list = []
for i in range(nms_image.shape[0]):
    for j in range(nms_image.shape[1]):
        if nms_image[i, j] > MAXThreshold:
            high_list.append((i, j))
direct = [(0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 0), (-1, 0), (0, -1)]


def DFS(stepmap, start):
    route = [start]
    while route:
        now = route.pop()
        if usemap[now] == 1:
            break
        usemap[now] = 1
        for dic in direct:
            next_coodinate = (now[0] + dic[0], now[1] + dic[1])
            if not usemap[next_coodinate] and nms_image[next_coodinate] > MINThreshold \
                    and stepmap.shape[0] - 1 > next_coodinate[0] >= 0 \
                    and stepmap.shape[1] - 1 > next_coodinate[1] >= 0:
                route.append(next_coodinate)


for i in high_list:
    DFS(nms_image, i)
plt.imshow(nms_image * usemap, cmap="gray")
plt.axis("off")
plt.show()


def CannyAlogrithm(image, MINThreshold, MAXThreshold):
    image = GaussianSmooth(image)
    Gx = Main.SobelAlogrithm(image, "horizontal")
    Gy = Main.SobelAlogrithm(image, "vertical")
    G = np.sqrt(np.square(Gx.astype(np.float64)) + np.square(Gy.astype(np.float64)))
    G = G * (255 / np.max(G)).astype(np.uint8)
    cita = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))
    nms_image = NonmaximumSuppression(G, cita)
    nms_image = (nms_image * (255 / np.max(nms_image))).astype(np.uint8)
    usemap = np.zeros(nms_image.shape)
    high_list = []
    for i in range(nms_image.shape[0]):
        for j in range(nms_image.shape[1]):
            if nms_image[i, j] > MAXThreshold:
                high_list.append((i, j))

    direct = [(0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 0), (-1, 0), (0, -1)]

    def DFS(stepmap, start):
        route = [start]
        while route:
            now = route.pop()
            if usemap[now] == 1:
                break
            usemap[now] = 1
            for dic in direct:
                next_coodinate = (now[0] + dic[0], now[1] + dic[1])
                if stepmap.shape[0] - 1 > next_coodinate[0] >= 0 \
                        and stepmap.shape[1] - 1 > next_coodinate[1] >= 0 \
                        and not usemap[next_coodinate] and nms_image[next_coodinate] > MINThreshold:
                    route.append(next_coodinate)

    for i in high_list:
        DFS(nms_image, i)
    return nms_image * usemap
