import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Đọc ảnh đường phố (ảnh gốc)
image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png")  # Đọc ảnh
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển thành RGB
height, width, _ = image.shape  # Lấy kích thước ảnh

# Đọc ảnh nhãn (label)
label_image = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\label\train1.png", cv2.IMREAD_GRAYSCALE)  # Đọc ảnh nhãn ở chế độ grayscale
label_image = cv2.resize(label_image, (width, height))  # Resize để trùng kích thước ảnh gốc

# Chuyển ảnh thành dữ liệu huấn luyện
features = []
labels = []

for y in range(height):
    for x in range(width):
        pixel = image[y, x]  # Lấy giá trị RGB tại pixel (y, x)
        feature_vector = [pixel[0], pixel[1], pixel[2], x / width, y / height]  # RGB + vị trí (x, y)
        features.append(feature_vector)
        labels.append(label_image[y, x])  # Gán nhãn từ ảnh label

# Chuyển danh sách thành numpy array
features = np.array(features)
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Khởi tạo mô hình Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)  # Huấn luyện mô hình

# Đánh giá mô hình
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Dự đoán phân đoạn ảnh mới
segmented_image = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        pixel = image[y, x]
        feature_vector = [[pixel[0], pixel[1], pixel[2], x / width, y / height]]
        pred_label = clf.predict(feature_vector)[0]
        segmented_image[y, x] = pred_label  # Gán nhãn dự đoán cho ảnh

# Hiển thị ảnh kết quả
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
