import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Hiển thị thanh tiến trình

# Thiết lập seed để tái lặp kết quả
np.random.seed(42)
# RGB to Class ID mapping (dựa trên yêu cầu của bạn)
color_to_class = {
    (143, 0, 1): 0,     # Xe
    (13, 223, 222): 1,  # Biển báo
    (34, 142, 106): 2,  # Cây
    (127, 63, 128): 3,  # Đường
    (70, 70, 70): 4,    # Tòa nhà
    (180, 130, 70): 5,  # Bầu trời
    (230, 35, 245): 6,  # Vỉa hè
}

class_names = {
    0: "Car",
    1: "Sign",
    2: "Tree",
    3: "Road",
    4: "Building",
    5: "Sky",
    6: "Sidewalk"
}
def extract_features(img, x, y, window_size=5):
    """
    Trích xuất đặc trưng cho một pixel tại vị trí (x, y)
    Đặc trưng gồm: RGB, HSV, vị trí (x,y), texture (LBP), gradient
    """
    # Lấy vùng lân cận (tránh việc ra khỏi biên ảnh)
    half = window_size // 2
    patch = img[max(0, y-half):y+half+1, max(0, x-half):x+half+1]
    
    # Đặc trưng màu sắc
    r, g, b = img[y, x]
    hsv = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_RGB2HSV)[0,0]
    
    # Đặc trưng vị trí (chuẩn hóa về [0, 1])
    height, width = img.shape[:2]
    norm_x, norm_y = x / width, y / height
    
    # Đặc trưng texture (LBP)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    else:
        lbp_hist = np.zeros(9)
    
    # Gradient (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # Kết hợp tất cả đặc trưng
    features = [
        r, g, b,                 # RGB
        hsv[0], hsv[1], hsv[2],  # HSV
        norm_x, norm_y,          # Vị trí
        *lbp_hist,               # LBP (9 bins)
        grad_mag                 # Gradient magnitude
    ]
    
    return np.array(features)

def load_data(img_path, label_path, num_images=100, img_size=(128, 256)):
    X = []
    y = []
    
    # Lấy danh sách ảnh (chỉ lấy 10 ảnh đầu)
    img_files = sorted(os.listdir(img_path))[:num_images]
    label_files = sorted(os.listdir(label_path))[:num_images]
    
    for img_file, label_file in tqdm(zip(img_files, label_files), total=num_images, desc="Processing images"):
        # Load ảnh và label
        img = cv2.imread(os.path.join(img_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV đọc ảnh BGR → chuyển sang RGB
        img = cv2.resize(img, img_size)
        
        label = cv2.imread(os.path.join(label_path, label_file))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, img_size)
        
        # Chuyển label từ RGB → Class ID
        label_class = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for color, class_id in color_to_class.items():
            mask = np.all(label == np.array(color), axis=-1)
            label_class[mask] = class_id
        
        # Trích xuất đặc trưng ngẫu nhiên 1000 pixel/ảnh (để giảm bộ nhớ)
        h, w = img.shape[:2]
        random_pixels = np.random.choice(h * w, size=min(1000, h * w), replace=False)
        
        for pixel in random_pixels:
            y_pixel = pixel // w
            x_pixel = pixel % w
            features = extract_features(img, x_pixel, y_pixel)
            X.append(features)
            y.append(label_class[y_pixel, x_pixel])
    
    return np.array(X), np.array(y)

# Đường dẫn dữ liệu
img_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\img"
label_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\label"
val_path_img = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\img"

# Load dữ liệu
X, y = load_data(img_path, label_path, num_images=100, img_size=(48, 128))
print("Shape of X:", X.shape)  # (n_samples, n_features)
print("Shape of y:", y.shape)  # (n_samples,)

from sklearn.model_selection import train_test_split

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Đánh giá trên tập test
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Tính confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Chuẩn hóa theo hàng (để xem recall)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=class_names.values(), yticklabels=class_names.values())
# plt.title('Confusion Matrix', fontsize=14)
# plt.xlabel('Dự đoán', fontsize=12)
# plt.ylabel('Thực tế', fontsize=12)
# plt.savefig('confusion_matrix.png')
# plt.show()


# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=class_names.values(), yticklabels=class_names.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix (Recall)')
plt.show()

# Lấy độ quan trọng của từng đặc trưng
feature_importance = rf.feature_importances_
feature_names = [
    'R', 'G', 'B', 'H', 'S', 'V', 
    'X', 'Y', 
    *[f'LBP_{i}' for i in range(9)], 
    'Grad_Mag'
]

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

def calculate_iou(y_true, y_pred, n_classes):
    iou_scores = []
    for class_id in range(n_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        iou = intersection / (union + 1e-6)  # Tránh chia 0
        iou_scores.append(iou)
    
    return np.mean(iou_scores)

miou = calculate_iou(y_test, y_pred, len(class_names))
print(f"mIoU: {miou:.4f}")

def predict_image(model, img_path, img_size=(96, 256)):
    # Load và tiền xử lý ảnh
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    
    # Dự đoán từng pixel
    h, w = img.shape[:2]
    mask_pred = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            features = extract_features(img, x, y)
            mask_pred[y, x] = model.predict([features])[0]
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_pred, cmap='jet', vmin=0, vmax=len(class_names)-1)
    plt.title('Predicted Segmentation')
    plt.colorbar(ticks=range(len(class_names)), label='Class ID')
    plt.show()

# Test trên ảnh mới
test_img_path = os.path.join(val_path_img, "val1.png")  # Thay đổi thành ảnh của bạn
predict_image(rf, test_img_path)
