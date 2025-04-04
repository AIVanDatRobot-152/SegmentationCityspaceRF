import cv2
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\real\pexels-photo-337909.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Khởi tạo SIFT
sift = cv2.SIFT_create()
        
# Thiết lập grid để trích xuất đặc trưng dày đặc (Dense)
step_size = 20  # Khoảng cách giữa các điểm
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray_image.shape[0], step_size)
      for x in range(0, gray_image.shape[1], step_size)]
        
# Trích xuất đặc trưng
_, des = sift.compute(gray_image, kp)  # des là vector đặc trưng SIFT
        
# Vẽ keypoints lên ảnh
sift_image = cv2.drawKeypoints(gray_image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Hiển thị ảnh gốc và ảnh với Dense SIFT features
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Ảnh gốc (xám)')

plt.subplot(1, 2, 2)
plt.imshow(sift_image, cmap='gray')
plt.title('Dense SIFT Features')
plt.show()

# In thông tin về các đặc trưng trích xuất được
if des is not None:
    print(f"Tổng số đặc trưng SIFT: {len(des)}")
    print(f"Kích thước mỗi đặc trưng: {des.shape[1]}")  # SIFT descriptor có 128 dimensions
else:
    print("Không trích xuất được đặc trưng nào")