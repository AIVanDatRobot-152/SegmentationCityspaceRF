import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage import exposure
import seaborn as sns
from time import time
from tqdm import tqdm
import joblib
import psutil

# Thiết lập giao diện đồ họa
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette('husl')

# Đường dẫn dữ liệu và model
img_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\img"
label_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\label"
MODEL_PATH = 'cityscapes_segmentation_model.joblib'

# Tham số
IMG_HEIGHT = 96
IMG_WIDTH = 256
NUM_SAMPLES = 100000  # Số pixel để huấn luyện
PIXELS_PER_IMAGE = IMG_HEIGHT * IMG_WIDTH

mem = psutil.virtual_memory()
if mem.available < 8 * 1024**3:  # < 8GB RAM
    NUM_SAMPLES = 50000  # Giảm mẫu nếu RAM thấp

def extract_features(img_path, label_path):
    """Trích xuất đặc trưng từ ảnh và nhãn phân đoạn"""
    # Đọc ảnh và chuyển sang grayscale
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {img_path}")
    
    # Đọc nhãn phân đoạn (ảnh grayscale)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Không thể đọc nhãn từ {label_path}")
    
    # Tiền xử lý ảnh
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_gray = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
    label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # Phát hiện cạnh Canny
    edges = cv2.Canny(img_gray, 100, 200)
    
    # Trích xuất đặc trưng HOG từ ảnh gốc
    fd_hog, hog_image = hog(img_gray, orientations=8, pixels_per_cell=(32, 32),
                           cells_per_block=(1, 1), visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # Trích xuất đặc trưng Dense SIFT
    sift = cv2.SIFT_create()
    kp = [cv2.KeyPoint(x, y, 20) for y in range(0, IMG_HEIGHT, 20) 
                               for x in range(0, IMG_WIDTH, 20)]
    _, dense_sift = sift.compute(img_gray, kp)
    if dense_sift is None:
        dense_sift = np.zeros((len(kp), 128))
    
    # Kết hợp các đặc trưng
    features = np.concatenate([
        img_gray.flatten()/255.0,
        edges.flatten()/255.0,
        hog_image.flatten(),
        fd_hog,
        dense_sift.flatten()
    ])
    
    return features, label

def process_pixel_data(images, labels):
    """Xử lý dữ liệu pixel-level"""
    pixel_features = []
    pixel_labels = []
    
    for img_features, label_img in zip(images, labels):
        if label_img.ndim == 1:
            label_img = label_img.reshape(IMG_HEIGHT, IMG_WIDTH)
        
        repeated_features = np.tile(img_features, (PIXELS_PER_IMAGE, 1))
        pixel_features.append(repeated_features)
        pixel_labels.append(label_img.flatten())
    
    # Kiểm tra kích thước các mảng
    print(f"Tổng số pixel: {sum(len(x) for x in pixel_labels)}")
    
    # Gộp tất cả các pixel lại
    X = np.concatenate(pixel_features)
    y = np.concatenate(pixel_labels)
    
    # Lấy mẫu ngẫu nhiên
    indices = np.random.choice(len(X), min(NUM_SAMPLES, len(X)), replace=False)
    return X[indices], y[indices]

# 1. Tải và xử lý dữ liệu
print("Đang tải và xử lý dữ liệu...")
start_time = time()
images, labels = [], []

# Lấy danh sách file ảnh và nhãn
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

# Đảm bảo số lượng ảnh và nhãn khớp nhau
assert len(img_files) == len(label_files), "Số lượng ảnh và nhãn không khớp"

# Xử lý từng cặp ảnh-nhãn
for img_name, label_name in tqdm(zip(img_files, label_files), total=len(img_files)):
    try:
        features, label = extract_features(
            os.path.join(img_dir, img_name),
            os.path.join(label_dir, label_name)
        )
        images.append(features)
        labels.append(label.flatten())
    except Exception as e:
        print(f"Lỗi khi xử lý {img_name}: {str(e)}")

images = np.array(images)
labels = np.array(labels)
print(f"Hoàn thành trong {time()-start_time:.2f} giây")
print(f"Tổng số mẫu: {len(images)}")
print(f"Kích thước dữ liệu đặc trưng: {images.shape}")
print(f"Kích thước nhãn: {labels.shape}")

# 2. Phân tích dữ liệu
# Đếm số pixel cho mỗi lớp trong tất cả các nhãn
unique_classes, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=unique_classes, y=counts)
plt.title('Phân bố pixel theo lớp')
plt.xlabel('Lớp')
plt.ylabel('Số lượng pixel (log scale)')
plt.yscale('log')
plt.savefig('class_distribution.png', bbox_inches='tight')
plt.close()

# 3. Chia tập dữ liệu (chọn ngẫu nhiên một số pixel để huấn luyện)
# Do dữ liệu phân đoạn rất lớn (pixel-level), chúng ta sẽ lấy mẫu
try:
    X, y = process_pixel_data(images, labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
except Exception as e:
    print(f"Lỗi khi xử lý dữ liệu: {str(e)}")
    exit()

# 4. Tối ưu siêu tham số
print("\nĐang tối ưu siêu tham số...")
param_grid = {'var_smoothing': np.logspace(-12, -2, 20)}

grid_search = GridSearchCV(
    GaussianNB(),
    param_grid=param_grid,
    cv=3,  # Giảm số fold để tăng tốc
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

# Kết quả tối ưu
best_nb = grid_search.best_estimator_
print(f"\nTham số tối ưu: {grid_search.best_params_}")
print(f"Độ chính xác tốt nhất: {grid_search.best_score_:.4f}")

# Biểu đồ hiệu suất theo siêu tham số
plt.figure(figsize=(12, 6))
plt.semilogx(param_grid['var_smoothing'], 
             grid_search.cv_results_['mean_test_score'], 
             marker='o', color='royalblue', linewidth=2)
plt.fill_between(param_grid['var_smoothing'],
                 grid_search.cv_results_['mean_test_score'] - grid_search.cv_results_['std_test_score'],
                 grid_search.cv_results_['mean_test_score'] + grid_search.cv_results_['std_test_score'],
                 alpha=0.2, color='royalblue')
plt.xlabel('Giá trị var_smoothing (log scale)')
plt.ylabel('Độ chính xác')
plt.title('Hiệu suất theo siêu tham số')
plt.grid(True, which="both", ls="--")
plt.savefig('hyperparameter_performance.png', bbox_inches='tight')
plt.close()

# 5. Đường cong học tập
best_nb = grid_search.best_estimator_
train_sizes, train_scores, test_scores = learning_curve(
    best_nb, X_train, y_train, cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),  # Giảm số điểm để tăng tốc
    scoring='accuracy'
)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Validation score')
plt.fill_between(train_sizes,
                 np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                 alpha=0.1, color='r')
plt.fill_between(train_sizes,
                 np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                 np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                 alpha=0.1, color='g')
plt.xlabel('Số lượng mẫu huấn luyện')
plt.ylabel('Độ chính xác')
plt.title('Đường cong học tập')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('learning_curve.png', bbox_inches='tight')
plt.close()

# 6. Đánh giá mô hình cuối cùng
y_pred = best_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác trên tập test: {accuracy*100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Ma trận nhầm lẫn (chỉ hiển thị một số lớp chính do có quá nhiều lớp)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
# Chỉ lấy 10 lớp phổ biến nhất để hiển thị
top_classes = np.argsort(-np.bincount(y_test.astype(int)))[:10]
sns.heatmap(cm[np.ix_(top_classes, top_classes)], annot=True, fmt='d', cmap='Blues')
plt.title('Ma trận nhầm lẫn (10 lớp phổ biến nhất)')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

best_nb.fit(X_train, y_train)
y_pred = best_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Lưu model
joblib.dump(best_nb, MODEL_PATH)
print(f"\nĐã lưu model vào {MODEL_PATH}")

# [Giữ nguyên phần đánh giá và hiển thị kết quả]

# Hàm để load model sau này
def load_segmentation_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

print("\nĐã hoàn thành quá trình huấn luyện và đánh giá!")

# 7. Hiển thị kết quả phân đoạn trên một số ảnh test
def display_segmentation_results(img_dir, label_dir, model, num_images=3):
    test_images = [f for f in os.listdir(img_dir) if f.endswith('.png')][:num_images]
    
    for img_name in test_images:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name)
        
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        
        # Tạo đặc trưng cho từng pixel
        h, w = img_gray.shape
        pixels = np.zeros((h*w, X_train.shape[1]))
        
        # Tính toán đặc trưng cho từng pixel (đơn giản hóa)
        for i in range(h):
            for j in range(w):
                # Trong thực tế cần tính toán đặc trưng cục bộ cho từng pixel
                pixels[i*w + j] = np.concatenate([
                    [img_gray[i,j]/255.0],
                    np.zeros(IMG_HEIGHT * IMG_WIDTH - 1),  # Sửa từ IMG_SIZE*IMG_SIZE
                    np.zeros(IMG_HEIGHT * IMG_WIDTH),    # HOG image
                    np.zeros(8),                     # HOG features
                    np.zeros(128)                    # Dense SIFT
                ])
        
        # Dự đoán
        pred_labels = model.predict(pixels)
        pred_mask = pred_labels.reshape(h, w)
        
        # Hiển thị kết quả
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        true_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        true_label = cv2.resize(true_label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        plt.imshow(true_label, cmap='jet')
        plt.title('Nhãn thực')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='jet')
        plt.title('Dự đoán')
        plt.axis('off')
        
        plt.suptitle(f'Kết quả phân đoạn cho {img_name}')
        plt.tight_layout()
        plt.savefig(f'segmentation_result_{img_name}.png', bbox_inches='tight')
        plt.close()

display_segmentation_results(img_dir.replace('train', 'val'), 
                            label_dir.replace('train', 'val'), 
                            best_nb)

print("\nĐã hoàn thành quá trình huấn luyện và đánh giá!")
print("Các biểu đồ đã được lưu vào thư mục hiện tại.")