import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo mô hình phân đoạn của Mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Đọc ảnh
image = cv2.imread(r"C:\Users\ACER\OneDrive\Pictures\oneday\2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

# Dự đoán mặt nạ phân đoạn
results = segment.process(image)

# Tạo mặt nạ nhị phân (1 là người, 0 là nền)
mask = results.segmentation_mask
binary_mask = (mask > 0.5).astype(np.uint8)  # Ngưỡng 0.5 để tách người khỏi cảnh

# Tạo ảnh tô màu
person_color = np.zeros_like(image)
person_color[:] = [0, 255, 0]  # Màu xanh lá cây cho người

background_color = np.zeros_like(image)
background_color[:] = [128, 0, 128]  # Màu tím cho nền

# 1️⃣ Ảnh sau phân đoạn (người màu xanh, nền màu tím)
segmented_image = person_color * binary_mask[:, :, None] + background_color * (1 - binary_mask[:, :, None])

# 2️⃣ Ảnh gốc với người được chồng màu xanh (giữ nền ban đầu)
overlay_image = image.copy()
overlay_image[binary_mask == 1] = (0.6 * image[binary_mask == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(segmented_image)
plt.title("Ảnh sau phân đoạn")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_image)
plt.title("chồng ảnh phân đoạn lên ảnh gốc")
plt.axis("off")

plt.show()
