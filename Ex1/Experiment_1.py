import cv2 as cv
import time
import numpy as np

def color_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的处理"""
    if img.shape[-1] != 3 and deal_Type != 1:
        pass
    if deal_Type == 1:
        # 手写图像取反色处理
        time1 = time.time()
        for row in range(img.shape[0]):  # 遍历每一行
            for col in range(img.shape[1]):  # 遍历每一列
                img[row][col] = 255 - img[row][col]
        time2 = time.time()
        print("数据检索遍历时间：", (time2 - time1) * 1000)
        # opencv求反色函数
        # time1 = time.time()
        # img = cv.bitwise_not(img)
        # time2 = time.time()
        # print("opencv遍历时间：", (time2 - time1) * 1000)

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

def img_sample(img, iv):
    new_img = np.zeros((int(img.shape[0] / iv), int(img.shape[1] / iv), 3), np.uint8)
    for rr in range(new_img.shape[0]):
            for cc in range(new_img.shape[1]):
                for val in range(new_img.shape[2]):
                    new_img[rr][cc][val] = np.uint8(np.mean(img[(rr*iv):((rr+1)* iv -1), (cc*iv):((cc+1)* iv -1), val]))
    return new_img

def img_quanty(img, q_Size):
    times = 256/q_Size
    new_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i][j] = np.uint8((round(img[i][j]/times))*times)
    return new_img