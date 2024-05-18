import cv2 as cv
import numpy as np
import math as m
from numba import jit # 转换为机器代码，加速运算
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment_3.Conv_UI import Ui_Dialog as UI_Conv
import time
from queue import PriorityQueue as pq

def to_256(val): # finished
    if val > 255:
        return 255
    elif val < 0:
        return 0
    return val

def edge_detect(img, deal_Type): #finished
    """根据用户的选择，对于图像做相应的灰度增强处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = prewitt(img)
    elif deal_Type == 2:
        img = sobel(img)
    elif deal_Type == 3:
        img = log(img)
    elif deal_Type == 4:
        img = canny(img)
    return img

def prewitt(img): # finished
    """*功能 : 根据prewitt对应的卷积模板，对图像进行边缘检测
    *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    prw_convs = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                          [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], dtype=int)
    # 图像遍历, 求取prewitt边缘
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_img[i][j] = to_256((np.abs(np.sum(img[i-1:i+2,j-1:j+2]*prw_convs[0])) + np.abs(np.sum(img[i-1:i+2,j-1:j+2]*prw_convs[1]))))
    time2 = time.time()  # 程序计时结束
    print("prewitt算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

    #cv程序编写
    x = cv.filter2D(img, cv.CV_16S, prw_convs[1])
    y = cv.filter2D(img, cv.CV_16S, prw_convs[0])
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    new_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    time2 = time.time()  # 程序计时结束
    print("prewitt算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

def sobel(img): # finished
    """*功能 : 根据sobel对应的卷积模板，对图像进行边缘检测
       *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    prw_convs = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=int)
    # 图像遍历, 求取sobel边缘
   
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_img[i][j] = to_256(np.round(m.sqrt((np.sum(img[i-1:i+2,j-1:j+2]*prw_convs[0])**2)+(np.sum(img[i-1:i+2,j-1:j+2]*prw_convs[1])**2))))
    
    time2 = time.time()  # 程序计时结束
    print("sobel算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

    # cv程序编写
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    new_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    time2 = time.time()  # 程序计时结束
    print("sobel算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

def log(img): # finished
    """*功能 : 根据LOG算子对应的卷积模板，对图像进行边缘检测"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
                          [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=int)
    # 图像遍历, 求取LOG边缘
    
    for i in range(2, rows-2):
        for j in range(2, cols-2):
            new_img[i][j] = to_256(np.sum(img[i-2:i+3,j-2:j+3]*prw_conv))
    time2 = time.time()  # 程序计时结束
    print("log算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # cv程序编写
    """这里写入你的程序"""
    
    # time2 = time.time()  # 程序计时结束
    # print("log算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

def canny(img): # finished
    """*功能 : canny算子
    *"""
    new_img = cv.Canny(img, 100, 200)

    return new_img

def otsu(img, jug): # finished
    """*功能：大津阈值分割，求取直方图数组，根据类间方差最大原理自动选择阈值，
    *注意：只处理灰度图像"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    hist, bins = np.histogram(img, np.arange(0,257))

    pixel_number = img.shape[0] * img.shape[1]
    mean_weigth = 1.0/pixel_number
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(hist[:t])
        pcf = np.sum(hist[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*hist[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*hist[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    new_img = img.copy()
    new_img[img > final_thresh] = 255
    new_img[img < final_thresh] = 0

    time2 = time.time()  # 程序计时结束
    print("大津阈值手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

   # opencv大津阈值分割
    max_t, new_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    # time2 = time.time()  # 程序计时结束
    # print("大津阈值cv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    if jug:
        plt.plot(hist, color="r", label="otsu value in histogram")
        plt.xlim([0, 256])
        plt.axvline(max_t, color='green') # 在直方图中绘制出阈值位置
        plt.legend() # 用于给图像加图例，各种符号和颜色所代表内容与指标的说明
        plt.show()
    return new_img

def hough_detect(img, deal_Type): # finished
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = line_detect(img, 3)
    elif deal_Type == 2:
        img = circle_detect(img)
    return img

def hough_transform(img): # finished
    """根据传入的图像，求取目标点对应的hough域，公式：ρ = x cos θ + y sin θ
    注：默认图像中255点为目标点"""
    rows, cols = img.shape[:2]  # 获取宽和高
    hg_rows, hg_cols = 180, int(m.sqrt(cols * cols + rows * rows))
    hough_img = np.zeros((hg_rows, hg_cols), dtype=np.int)  # 新建hough域，全为0值

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 0:
                for theta in range(180):
                    hough_img[theta][round(i*m.cos(theta/180*m.pi) + j*m.sin(theta/180*m.pi))] += 1
    return hough_img

def line_detect(img, num): #finished
    """*功能 : 通过hough变换检测直线，num:需检测直线的条数"""
    time1 = time.time()  # 程序计时开始
    hough_img = hough_transform(img)
    rows, cols = img.shape[:2]  # 获取原图宽和高

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for times in range(num):
        theta, rho = np.where(hough_img == np.max(hough_img))
        thetaMax = theta[0]
        rhoMax = rho[0]
        hough_img[thetaMax][rhoMax] = 0
        pt_start = (round(rhoMax/m.sin(thetaMax/180*m.pi)), 0)
        pt_end = (round((rhoMax-cols*m.cos(thetaMax/180*m.pi))/m.sin(thetaMax/180*m.pi)),cols)
        cv.line(img, pt_start, pt_end, (0, 0, 255), 1)
    time2 = time.time() # 程序计时结束
    print("hough直线检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return img

    # opencv函数检测直线
    lines = cv.HoughLines(img, 1, np.pi / 180, 100)  # 这里对最后一个参数使用了经验型的值
    hough_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) # 转彩色，方便显示
    for line in lines:
        p, a = line[0]  # 第一个元素是距离rho, 第二个元素是角度theta
        pt_start = (0, int(p/m.sin(a))) # 绘制直线起点
        pt_end = (cols, int((p-cols*m.cos(a))/m.sin(a))) # 绘制直线终点
        cv.line(hough_img, pt_start, pt_end, (0, 0, 255), 1)
    time2 = time.time() # 程序计时结束
    print("hough直线检测opencv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    return img, hough_img

def circle_detect(img):
    """*功能 : 直接利用opencv中的hough圆检测，检测出图像中的圆"""
    time1 = time.time()  # 程序计时开始
    # 霍夫变换圆检测
   
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 300, param1=100, param2=30, minRadius=0, maxRadius=0)
    for i in circles[0, :]:  # 遍历矩阵每一行的数据
        
        cv.circle(img, (round(i[0]), round(i[1])), round(i[2]), (0, 255, 0), 2)
        cv.circle(img, (round(i[0]), round(i[1])), 2, (0, 0, 255), 3)
    time2 = time.time()  # 程序计时结束
    print("hough圆检测opencv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return img
