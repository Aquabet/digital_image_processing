import cv2 as cv
import numpy as np
import math as m
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment_3.Linear_UI import Ui_Dialog as UI_Linear
from Experiment_3.Log_UI import Ui_Dialog as UI_Log
from Experiment_3.Exp_UI import Ui_Dialog as UI_Exp
from Experiment_3.Pow_UI import Ui_Dialog as UI_Pow
from Experiment_3.Conv_UI import Ui_Dialog as UI_Conv
import time


def gray_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰度增强处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = linear_strench(img)
    elif deal_Type == 2:
        img = log_strench(img)
    elif deal_Type == 3:
        img = exp_strench(img)
    elif deal_Type == 4:
        img = pow_strench(img)
    return img


def linear_strench(img):
    """*功能 : 根据传入的图像及给定的c,d两个灰值区间参数值，进行线性拉伸
    *注意，只对灰度图像拉伸，函数：g(x,y)=(d-c)/(b-a)*[f(x,y)-a]+c=k*[f(x,y)-a]+c"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    a, b = np.min(img), np.max(img)
    q_dialog = QDialog()
    dialog = UI_Linear()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(a))  # 显示原图灰度范围
    dialog.lineEdit_b.setText(str(b))
    dialog.lineEdit_c.setText(str(a))  # 初始化变换后灰度范围
    dialog.lineEdit_d.setText(str(b))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互变换后灰度范围
        c = int(dialog.lineEdit_c.text())
        d = int(dialog.lineEdit_d.text())
        time1 = time.time()  # 程序计时开始
       # 这里写入你的核心代码
        for i in range(rows):
            for j in range(cols):
                new_img[i][j] = (d-c)/(b-a)*(img[i][j]-a)+c

        time2 = time.time()  # 程序计时结束
        print("灰度增强程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        return new_img


def log_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行对数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=a+lg[f(x,y)+1]/(c*lgb)"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Log()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(0.0))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(2.0))
    dialog.lineEdit_c.setText(str(0.03))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(
            dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if c == 0 or b <= 0 or b == 1:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time()  # 程序计时开始

        # 这里写入你的核心代码
        for i in range(rows):
            for j in range(cols):
                new_img[i][j] = a+m.log10(img[i][j]+1)/(c*m.log10(b))

        time2 = time.time()  # 程序计时结束
        print("对数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


def exp_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行指数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=b^c[f(x,y)-a]-1"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Exp()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(150))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(1.5))
    dialog.lineEdit_c.setText(str(0.6))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(
            dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if b <= 0 or b == 1:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time()  # 程序计时开始

        # 这里写入你的核心代码
        for i in range(rows):
            for j in range(cols):
                new_img[i][j] = (b**c)*(img[i][j]-a)-1

        time2 = time.time()  # 程序计时结束
        print("指数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


def pow_strench(img):
    """*功能 : 根根据传入的图像及给定的c,r两个参数值，进行幂律非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=c[f(x,y)]^r"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Pow()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_c.setText(str(1))  # 初始化对数变换参数
    dialog.lineEdit_r.setText(str(1.5))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        c, r = float(dialog.lineEdit_c.text()), float(dialog.lineEdit_r.text())
        if r <= 0 or c <= 0:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time()  # 程序计时开始

        # 这里写入你的核心代码
        for i in range(rows):
            for j in range(cols):
                new_img[i][j] = c*(img[i][j]**r)

        time2 = time.time()  # 程序计时结束
        print("幂律变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


def hist_equalization(img, jug):
    """*功能 : 直方图均衡化算法, jug判断返回是图像/直方图"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大一9致的空图像
    time1 = time.time()  # 程序计时开始
    hist = creat_histogram(img)
    # 这里写入你的核心代码
    cnt = {}
    for i in range(256):
        cnt[i] = 0

    for i in range(rows):
        for j in range(cols):
            cnt[img[i][j]] += 1

    pixel = sum(cnt.values())

    for k in cnt.keys():
        cnt[k] /= pixel

    sumset = {}
    sumnum = 0

    for i in range(256):
        sumnum += cnt[i]
        sumset[i] = sumnum

    for i in sumset:
        sumset[i] = int(255*sumset[i]+0.5)

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            new_img[i][j] = sumset[img[i][j]]

    time2 = time.time()  # 程序计时结束
    print("图像均衡算法程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    if jug:
        imgs = [img, new_img]
        colors = ("b", "r")
        texts = ("original histogram", "histogram after equalization")
        for i in range(2):
            hist = cv.calcHist([imgs[i]], [0], None, [256], [0, 255])
            plt.plot(hist, color=colors[i], label=texts[i])
        plt.xlim([0, 256])
        plt.legend()
        plt.show()
    return new_img


def creat_histogram(img):
    """*功能 : 计算传入图像直方图，若是彩色图像，计算各颜色分量直方图并返回"""
    rows, cols = img.shape[:2]  # 获取宽和高
    hist = []
    if img.ndim == 2:  # 灰度图像统计直方图
        hist = [0] * 256  # 建立灰度图像直方图
        # 图像遍历
        for row in range(rows):
            for col in range(cols):
                hist[img[row][col]] += 1
    elif img.ndim == 3:  # 彩色图像统计直方图
        hist = [[0] * 256, [0] * 256, [0] * 256]  # 建立彩色图像直方图
        # 图像遍历
        for row in range(rows):
            for col in range(cols):
                hist[0][img[row][col][0]] += 1
                hist[1][img[row][col][1]] += 1
                hist[2][img[row][col][2]] += 1
    return hist


def gray_smooth(img, deal_Type):
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = neighbor_average(img)
    elif deal_Type == 2:
        img = median_filter(img)
    return img


def neighbor_average(img):
    """*功能 : 用户交互卷积模板，获取卷积系数进行邻域平滑，只对灰度图像处理"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Conv()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array([[float(dialog.lineEdit1.text()), float(dialog.lineEdit2.text()), float(dialog.lineEdit3.text())],
                             [float(dialog.lineEdit4.text()), float(
                                 dialog.lineEdit5.text()), float(dialog.lineEdit6.text())],
                             [float(dialog.lineEdit7.text()), float(dialog.lineEdit8.text()), float(dialog.lineEdit9.text())]])
        np_kernel = np_kernel/np_kernel.sum()  # 正则化

        time1 = time.time()  # 程序计时开始

        # 这里写入你的核心代码
        for i in range(rows):
            for j in range(cols):
                if(i == 0 or i == rows-1 or j == 0 or j == cols-1):
                    new_img[i][j] = img[i][j]
                else:
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            new_img[i][j] += img[i+m][j+n]*np_kernel[m+1][n+1]

        time2 = time.time()  # 程序计时结束
        print("邻域平均平滑程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


def median_filter(img):
    """*功能 : 中值滤波"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    len = 3  # 定义中值滤波模板3×3

    time1 = time.time()  # 程序计时开始

    # 这里写入你的核心代码
    for i in range(rows):
        for j in range(cols):
            if(i == 0 or i == rows-1 or j == 0 or j == cols-1):
                new_img[i][j] = img[i][j]
            else:
                new_img[i][j] = np.median(img[i-1:i+2, j-1:j+2])

    time2 = time.time()  # 程序计时结束
    print("中值滤波程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
