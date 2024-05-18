# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainDetect_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1419, 707)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_AftShow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_AftShow.setGeometry(QtCore.QRect(710, 10, 690, 490))
        self.groupBox_AftShow.setObjectName("groupBox_AftShow")
        self.label_AftPicShow = QtWidgets.QLabel(self.groupBox_AftShow)
        self.label_AftPicShow.setGeometry(QtCore.QRect(10, 20, 670, 460))
        self.label_AftPicShow.setText("")
        self.label_AftPicShow.setObjectName("label_AftPicShow")
        self.groupBox_PreShow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_PreShow.setGeometry(QtCore.QRect(10, 10, 690, 490))
        self.groupBox_PreShow.setObjectName("groupBox_PreShow")
        self.label_PrePicShow = QtWidgets.QLabel(self.groupBox_PreShow)
        self.label_PrePicShow.setGeometry(QtCore.QRect(10, 20, 670, 460))
        self.label_PrePicShow.setText("")
        self.label_PrePicShow.setScaledContents(True)
        self.label_PrePicShow.setObjectName("label_PrePicShow")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 510, 381, 141))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label_datatype_2 = QtWidgets.QLabel(self.groupBox)
        self.label_datatype_2.setGeometry(QtCore.QRect(10, 30, 81, 20))
        self.label_datatype_2.setObjectName("label_datatype_2")
        self.comboBox_ColorDeal = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_ColorDeal.setGeometry(QtCore.QRect(100, 30, 121, 22))
        self.comboBox_ColorDeal.setObjectName("comboBox_ColorDeal")
        self.comboBox_ColorDeal.addItem("")
        self.comboBox_ColorDeal.setItemText(0, "")
        self.comboBox_ColorDeal.addItem("")
        self.comboBox_ColorDeal.addItem("")
        self.comboBox_ColorDeal.addItem("")
        self.comboBox_ColorDeal.addItem("")
        self.comboBox_ColorDeal.addItem("")
        self.slider_ImgSample = QtWidgets.QSlider(self.groupBox)
        self.slider_ImgSample.setGeometry(QtCore.QRect(90, 60, 231, 22))
        self.slider_ImgSample.setMinimum(1)
        self.slider_ImgSample.setMaximum(10)
        self.slider_ImgSample.setSingleStep(1)
        self.slider_ImgSample.setOrientation(QtCore.Qt.Horizontal)
        self.slider_ImgSample.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_ImgSample.setTickInterval(1)
        self.slider_ImgSample.setObjectName("slider_ImgSample")
        self.label_datatype_3 = QtWidgets.QLabel(self.groupBox)
        self.label_datatype_3.setGeometry(QtCore.QRect(10, 60, 81, 20))
        self.label_datatype_3.setObjectName("label_datatype_3")
        self.label_sample = QtWidgets.QLabel(self.groupBox)
        self.label_sample.setGeometry(QtCore.QRect(330, 60, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_sample.setFont(font)
        self.label_sample.setObjectName("label_sample")
        self.slider_ImgQuanty = QtWidgets.QSlider(self.groupBox)
        self.slider_ImgQuanty.setGeometry(QtCore.QRect(90, 100, 231, 22))
        self.slider_ImgQuanty.setMinimum(1)
        self.slider_ImgQuanty.setMaximum(100)
        self.slider_ImgQuanty.setSingleStep(5)
        self.slider_ImgQuanty.setProperty("value", 100)
        self.slider_ImgQuanty.setOrientation(QtCore.Qt.Horizontal)
        self.slider_ImgQuanty.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_ImgQuanty.setTickInterval(5)
        self.slider_ImgQuanty.setObjectName("slider_ImgQuanty")
        self.label_datatype_4 = QtWidgets.QLabel(self.groupBox)
        self.label_datatype_4.setGeometry(QtCore.QRect(10, 100, 81, 20))
        self.label_datatype_4.setObjectName("label_datatype_4")
        self.label_quanty = QtWidgets.QLabel(self.groupBox)
        self.label_quanty.setGeometry(QtCore.QRect(330, 100, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_quanty.setFont(font)
        self.label_quanty.setObjectName("label_quanty")
        self.label_datatype = QtWidgets.QLabel(self.centralwidget)
        self.label_datatype.setGeometry(QtCore.QRect(860, 630, 121, 20))
        self.label_datatype.setObjectName("label_datatype")
        self.comboBox_SelectData = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_SelectData.setGeometry(QtCore.QRect(980, 621, 121, 31))
        self.comboBox_SelectData.setObjectName("comboBox_SelectData")
        self.comboBox_SelectData.addItem("")
        self.comboBox_SelectData.setItemText(0, "")
        self.comboBox_SelectData.addItem("")
        self.comboBox_SelectData.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1419, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.comboBox_SelectData.currentIndexChanged['int'].connect(MainWindow.onbuttonclick_selectDataType)
        self.comboBox_ColorDeal.currentIndexChanged['int'].connect(MainWindow.oncombox_selectColorType)
        self.slider_ImgSample.valueChanged['int'].connect(MainWindow.onslide_imgSample)
        self.slider_ImgQuanty.valueChanged['int'].connect(MainWindow.onslide_imgQuanty)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_AftShow.setTitle(_translate("MainWindow", "测试实例"))
        self.groupBox_PreShow.setTitle(_translate("MainWindow", "输入实例"))
        self.groupBox.setTitle(_translate("MainWindow", "实验一"))
        self.label_datatype_2.setText(_translate("MainWindow", "色彩处理："))
        self.comboBox_ColorDeal.setItemText(1, _translate("MainWindow", "反色处理"))
        self.comboBox_ColorDeal.setItemText(2, _translate("MainWindow", "灰值化"))
        self.comboBox_ColorDeal.setItemText(3, _translate("MainWindow", "Lab颜色模型"))
        self.comboBox_ColorDeal.setItemText(4, _translate("MainWindow", "YCrCb颜色模型"))
        self.comboBox_ColorDeal.setItemText(5, _translate("MainWindow", "HSI颜色模型"))
        self.label_datatype_3.setText(_translate("MainWindow", "采样间隔："))
        self.label_sample.setText(_translate("MainWindow", "1"))
        self.label_datatype_4.setText(_translate("MainWindow", "量化范围："))
        self.label_quanty.setText(_translate("MainWindow", "0-100"))
        self.label_datatype.setText(_translate("MainWindow", "选择数据类型："))
        self.comboBox_SelectData.setItemText(1, _translate("MainWindow", "读入图像"))
        self.comboBox_SelectData.setItemText(2, _translate("MainWindow", "读入本地视频"))
