import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc
image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\lena.tif", cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám

# Kiểm tra nếu ảnh không được tải đúng
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn tệp!")

# Áp dụng ngưỡng để phân đoạn ảnh (Thresholding)
threshold_value = 127  # Giá trị ngưỡng
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Hiển thị ảnh gốc và ảnh sau khi phân đoạn
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Ảnh gốc")

plt.subplot(1,2,2)
plt.imshow(binary_image, cmap="gray")
plt.title("Ảnh sau khi phân đoạn")

plt.show()
