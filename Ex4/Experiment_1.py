import cv2 as cv
import numpy as np
import time
from numba import jit

def color_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的色彩变换处理"""
    if img.shape[-1] != 3 and deal_Type != 1:
        pass
    if deal_Type == 1:
        img = inverse_color(img)
    elif deal_Type == 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif deal_Type == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    elif deal_Type == 4:
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    elif deal_Type == 5:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    else:
        pass
    return img


@jit(nopython=True)
def inverse_color_sub(img):
    for row in range(img.shape[0]):  # 遍历每一行
        for col in range(img.shape[1]):  # 遍历每一列
            img[row][col] = 255 - img[row][col]
    return img


def inverse_color(img):
    time1 = time.time()
    img = inverse_color_sub(img)
    time2 = time.time()
    print("数据检索遍历时间： ", (time2 - time1) * 1000)

    # # opencv求反色函数
    # time1 = time.time()
    # img = cv.bitwise_not(img)
    # time2 = time.time()
    # cv.imwrite('bitwise.bmp', img)
    # print("opencv遍历时间：", (time2 - time1) * 1000)
    return img


def img_sample(img, iv):
    """根据传入的采样间隔参数，进行图像采样,
    *img:传入带采样图像; iv(interval):采样间隔参数"""
    new_img = img[::iv, ::iv, :] if img.shape[-1] == 3 else img[::iv, ::iv]
    return new_img


def img_quanty(img, q_Size):
    """根据传入的量化值，进行图像量化,
    *img:传入带采样图像; q_Size:量化范围"""
    time1 = time.time()  # 程序计时开始

    div = round(255 / q_Size)
    new_img = np.clip(np.floor(img / div) * div, 0, 255)
    new_img = new_img.astype(np.uint8)

    time2 = time.time()  # 程序计时结束
    print("量化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img