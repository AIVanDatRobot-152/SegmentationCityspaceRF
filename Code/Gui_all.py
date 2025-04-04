from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2
from skimage.feature import hog
from skimage import exposure

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
        self.label_6 = QLabel(self.InPut)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 20, 791, 581))
        self.tabWidget.addTab(self.InPut, "")
        self.hog = QWidget()
        self.hog.setObjectName(u"hog")
        self.label_5 = QLabel(self.hog)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 70, 791, 531))
        self.hogButton = QPushButton(self.hog)
        self.hogButton.setObjectName(u"hogButton")
        self.hogButton.setGeometry(QRect(320, 10, 171, 41))
        self.tabWidget.addTab(self.hog, "")
        self.sift = QWidget()
        self.sift.setObjectName(u"sift")
        self.label_7 = QLabel(self.sift)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(10, 60, 791, 531))
        self.siftButton = QPushButton(self.sift)
        self.siftButton.setObjectName(u"siftButton")
        self.siftButton.setGeometry(QRect(320, 10, 171, 41))
        self.tabWidget.addTab(self.sift, "")
        self.label = QWidget()
        self.label.setObjectName(u"label")
        self.label_8 = QLabel(self.label)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(10, 70, 791, 531))
        self.labelButton = QPushButton(self.label)
        self.labelButton.setObjectName(u"labelButton")
        self.labelButton.setGeometry(QRect(320, 10, 171, 41))
        self.tabWidget.addTab(self.label, "")
        self.output = QWidget()
        self.output.setObjectName(u"output")
        self.label_9 = QLabel(self.output)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(10, 70, 791, 531))
        self.Segmentation = QPushButton(self.output)
        self.Segmentation.setObjectName(u"Segmentation")
        self.Segmentation.setGeometry(QRect(320, 10, 171, 41))
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

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.browse.setText(QCoreApplication.translate("MainWindow", u"T\u1ea3i h\u00ecnh \u1ea3nh", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ff5921;\">Tr\u01b0\u1eddng \u0110\u1ea1i h\u1ecdc C\u00f4ng Ngh\u1ec7 TP.HCM (HUTECH )</span></p><p align=\"center\"><span style=\" font-size:12pt; color:#00aa7f;\">L\u1edbp: 22DRTA1 - Robot v\u00e0 Tr\u00ed tu\u1ec7 Nh\u00e2n t\u1ea1o</span><span style=\" font-size:16pt; font-weight:600; color:#ff5921;\"><br/></span><span style=\" font-size:18pt; font-weight:600; color:#00aa7f; vertical-align:sub;\">Nguy\u1ec5n V\u0103n \u0110\u1ea1t - 2286300010</span></p></body></html>", None))
        self.label_3.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.InPut), QCoreApplication.translate("MainWindow", u"\u1ea2nh g\u1ed1c", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.hogButton.setText(QCoreApplication.translate("MainWindow", u"HOG", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.hog), QCoreApplication.translate("MainWindow", u"HOG", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.siftButton.setText(QCoreApplication.translate("MainWindow", u"Dense SIFT", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sift), QCoreApplication.translate("MainWindow", u"Dense Sift", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.labelButton.setText(QCoreApplication.translate("MainWindow", u"Original Label", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.label), QCoreApplication.translate("MainWindow", u"Original Label", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.Segmentation.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.output), QCoreApplication.translate("MainWindow", u"\u1ea2nh sau ph\u00e2n \u0111o\u1ea1n", None))
        self.gray.setText(QCoreApplication.translate("MainWindow", u"Chuy\u1ec3n sang \u1ea3nh x\u00e1m", None))
    # retranslateUi

class ConsoleMainWindow(QMainWindow):
    def __init__(self):
        super(ConsoleMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.current_image = None

        # Kết nối các nút bấm
        self.ui.browse.clicked.connect(self.load_image)
        self.ui.gray.clicked.connect(self.convert_to_grayscale)
        self.ui.hogButton.clicked.connect(self.extract_hog_features)
        self.ui.siftButton.clicked.connect(self.extract_sift_features)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn hình ảnh",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)"
        )

        if not file_path:
            return

        # Đọc hình ảnh và hiển thị
        self.current_image = cv2.imread(file_path)
        if self.current_image is not None:
            # Chuyển ảnh sang RGB vì OpenCV sử dụng BGR
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.label_6.setPixmap(pixmap.scaled(self.ui.label_6.size(), Qt.KeepAspectRatio))

    def convert_to_grayscale(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có hình ảnh được tải.")
            return

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        bytes_per_line = width
        q_image = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_6.setPixmap(pixmap.scaled(self.ui.label_6.size(), Qt.KeepAspectRatio))

    def extract_hog_features(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có hình ảnh được tải.")
            return
        
        # Chuyển sang ảnh xám
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image

        # Tính HOG
        fd, hog_image = hog(
            gray_image,
            orientations=12,
            pixels_per_cell=(32, 32),
            cells_per_block=(1, 1),
            visualize=True,
            feature_vector=True
        )

        # Chuẩn hóa ảnh HOG và chuyển về [0, 255]
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)

        # Hiển thị bằng QLabel
        height, width = hog_image_rescaled.shape
        q_image = QImage(hog_image_rescaled.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_5.setPixmap(pixmap.scaled(self.ui.label_5.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def extract_sift_features(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có hình ảnh được tải.")
            return
        
        # Chuyển sang ảnh xám nếu cần
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image

        # Khởi tạo Dense SIFT
        sift = cv2.SIFT_create()
        
        # Thiết lập grid để trích xuất đặc trưng dày đặc (Dense)
        step_size = 20  # Khoảng cách giữa các điểm
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray_image.shape[0], step_size)
                                        for x in range(0, gray_image.shape[1], step_size)]
        
        # Trích xuất đặc trưng
        _, des = sift.compute(gray_image, kp)
        
        # Vẽ keypoints lên ảnh
        sift_image = cv2.drawKeypoints(gray_image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Chuyển đổi ảnh để hiển thị trong PyQt
        height, width = sift_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(sift_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_7.setPixmap(pixmap.scaled(self.ui.label_7.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwin = ConsoleMainWindow()
    mainwin.show()
    sys.exit(app.exec_())