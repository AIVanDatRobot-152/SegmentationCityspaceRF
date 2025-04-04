import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\lena.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị với matplotlib

# Chuyển ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Xác định ngưỡng cho màu đỏ
lower_red1 = np.array([0, 120, 70])  # Ngưỡng thấp (đỏ sẫm)
upper_red1 = np.array([10, 255, 255])  # Ngưỡng cao (đỏ sẫm)

lower_red2 = np.array([170, 120, 70])  # Ngưỡng thấp (đỏ sáng)
upper_red2 = np.array([180, 255, 255])  # Ngưỡng cao (đỏ sáng)

# Tạo mặt nạ để nhận diện vùng màu đỏ
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # Kết hợp hai mặt nạ

# Áp dụng mặt nạ lên ảnh gốc để chỉ giữ lại vùng màu đỏ
result = cv2.bitwise_and(image, image, mask=mask)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Ảnh gốc")

plt.subplot(2, 2, 2)
plt.imshow(hsv)
plt.title(" không gian màu HSV")

plt.subplot(2, 2, 3)
plt.imshow(mask, cmap="gray")
plt.title("Mặt nạ màu đỏ")

plt.subplot(2, 2, 4)
plt.imshow(result)
plt.title("Ảnh sau phân đoạn")

plt.show()
