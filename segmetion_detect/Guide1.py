# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Gui1BdiioS.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1320, 856)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.browse = QPushButton(self.centralwidget)
        self.browse.setObjectName(u"browse")
        self.browse.setGeometry(QRect(840, 190, 221, 81))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(290, 0, 731, 121))
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 0, 241, 71))
        self.label_3.setPixmap(QPixmap(u"C:/Users/ACER/OneDrive/Pictures/PNG/Hutech_logo.png"))
        self.label_3.setScaledContents(True)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(14, 169, 821, 641))
        self.InPut = QWidget()
        self.InPut.setObjectName(u"InPut")
        self.tabWidget.addTab(self.InPut, "")
        self.hog = QWidget()
        self.hog.setObjectName(u"hog")
        self.tabWidget.addTab(self.hog, "")
        self.sift = QWidget()
        self.sift.setObjectName(u"sift")
        self.tabWidget.addTab(self.sift, "")
        self.label = QWidget()
        self.label.setObjectName(u"label")
        self.tabWidget.addTab(self.label, "")
        self.output = QWidget()
        self.output.setObjectName(u"output")
        self.tabWidget.addTab(self.output, "")
        self.gray = QPushButton(self.centralwidget)
        self.gray.setObjectName(u"gray")
        self.gray.setGeometry(QRect(840, 270, 221, 81))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1320, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.browse.setText(QCoreApplication.translate("MainWindow", u"T\u1ea3i h\u00ecnh \u1ea3nh", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ff5921;\">Tr\u01b0\u1eddng \u0110\u1ea1i h\u1ecdc C\u00f4ng Ngh\u1ec7 TP.HCM (HUTECH )</span></p><p align=\"center\"><span style=\" font-size:12pt; color:#00aa7f;\">L\u1edbp: 22DRTA1 - Robot v\u00e0 Tr\u00ed tu\u1ec7 Nh\u00e2n t\u1ea1o</span><span style=\" font-size:16pt; font-weight:600; color:#ff5921;\"><br/></span><span style=\" font-size:18pt; font-weight:600; color:#00aa7f; vertical-align:sub;\">Nguy\u1ec5n V\u0103n \u0110\u1ea1t - 2286300010</span></p></body></html>", None))
        self.label_3.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.InPut), QCoreApplication.translate("MainWindow", u"\u1ea2nh g\u1ed1c", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.hog), QCoreApplication.translate("MainWindow", u"HOG", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sift), QCoreApplication.translate("MainWindow", u"Dense Sift", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.label), QCoreApplication.translate("MainWindow", u"Original Label", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.output), QCoreApplication.translate("MainWindow", u"\u1ea2nh sau ph\u00e2n \u0111o\u1ea1n", None))
        self.gray.setText(QCoreApplication.translate("MainWindow", u"Chuy\u1ec3n sang \u1ea3nh x\u00e1m", None))
    # retranslateUi



class ConsoleMainWindow(QMainWindow):
    def __init__(self):
        super(ConsoleMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwin = ConsoleMainWindow()
    mainwin.show()
    sys.exit(app.exec_())