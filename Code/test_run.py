import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score

# Cấu hình hiển thị
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette('husl')

# Đường dẫn dữ liệu
img_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\img"
label_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\label"
model_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\trained_model2.pkl"

# Kiểm tra sự tồn tại của mô hình
if not os.path.exists(model_path):
    raise FileNotFoundError("Không tìm thấy mô hình đã train! Hãy chạy lại quá trình train để tạo file .pkl")

# Tải mô hình
print("Loading trained model...")
clf = joblib.load(model_path)

# Lấy danh sách file ảnh và nhãn
img_files = sorted(os.listdir(img_path_folder))
label_files = sorted(os.listdir(label_path_folder))
if len(img_files) != len(label_files):
    raise ValueError("Số lượng ảnh và nhãn không khớp!")

# Định nghĩa hàm tiền xử lý ảnh
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {img_path}")
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Giữ nguyên kích thước ảnh 96x256
    fd, hog_image = hog(img, orientations=10, pixels_per_cell=(16, 16), 
                        cells_per_block=(1, 1), visualize=True)
    canny_edges = cv2.Canny(img, 100, 200).flatten()
    
    features = np.concatenate([img.flatten()/255.0, fd, canny_edges])
    return features

# Load dữ liệu
print("Đang tải và xử lý dữ liệu...")
X_test, y_test = [], []
for img_file, label_file in zip(img_files, label_files):
    img_path = os.path.join(img_path_folder, img_file)
    label_path = os.path.join(label_path_folder, label_file)
    
    X_test.append(preprocess_image(img_path))
    y_test.append(preprocess_image(label_path))

X_test = np.array(X_test)
y_test = np.array(y_test)

# Đảm bảo số lượng đặc trưng đầu vào khớp với mô hình
expected_features = clf.n_features_in_
if X_test.shape[1] != expected_features:
    raise ValueError(f"Số lượng đặc trưng không khớp! Model yêu cầu {expected_features}, nhưng dữ liệu có {X_test.shape[1]}")

# Dự đoán
print("Đang dự đoán...")
y_pred = clf.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='macro')
recall = recall_score(y_test.flatten(), y_pred.flatten(), average='macro')
precision = precision_score(y_test.flatten(), y_pred.flatten(), average='macro')

print(f"\nĐộ chính xác: {accuracy*100:.2f}%")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test.flatten(), y_pred.flatten()))

# Hiển thị các chỉ số lên biểu đồ
metrics = {
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'Accuracy': accuracy
}

# Vẽ biểu đồ thanh (bar chart) cho các chỉ số
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=['royalblue', 'green', 'orange', 'purple'])
plt.title('Các chỉ số đánh giá mô hình')
plt.xlabel('Chỉ số')
plt.ylabel('Giá trị')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.savefig('confusion_matrix.png')
plt.show()

# Hiển thị dự đoán mẫu
def display_predictions(X_test, y_test, y_pred, num_images=6):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(3, num_images, i+1)
        plt.imshow(X_test[i][:96*256].reshape(96, 256), cmap='gray')
        plt.title("Ảnh gốc")
        plt.axis('off')
        
        plt.subplot(3, num_images, i+1+num_images)
        plt.imshow(y_test[i][:96*256].reshape(96, 256), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(3, num_images, i+1+2*num_images)
        plt.imshow(y_pred[i][:96*256].reshape(96, 256), cmap='gray')
        plt.title("Dự đoán")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

display_predictions(X_test, y_test, y_pred)
