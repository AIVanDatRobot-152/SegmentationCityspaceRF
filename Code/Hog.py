import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\googleimg\traffic-jam-vehicles-highway.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Trích xuất HOG features và ảnh HOG
fd, hog_image = hog(
    gray_image,
    orientations=4,           
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),  
    visualize=True,
    feature_vector=True
)

# Chuẩn hóa ảnh HOG để hiển thị rõ hơn
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# hog_image_rescaled = exposure.equalize_hist(hog_image_rescaled)

# Hiển thị ảnh gốc và ảnh HOG
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Ảnh gốc (xám)')

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('Ảnh HOG')
plt.show()