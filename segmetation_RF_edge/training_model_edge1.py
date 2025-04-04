import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, feature, color, transform, filters
from skimage.feature import hog, daisy
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import os
import joblib
from tqdm import tqdm
import time

# Đường dẫn ảnh và nhãn
img_path_folder = r"E:\NguyenVanDat_22DRTA1\train\img"
label_path_folder = r"E:\NguyenVanDat_22DRTA1\train\label"
model_path = "trained_model_edge.pkl"
features_path = "extracted_features.npz"

# Lấy danh sách file ảnh và nhãn
img_files = sorted(os.listdir(img_path_folder))
label_files = sorted(os.listdir(label_path_folder))
if len(img_files) != len(label_files):
    raise ValueError("Số lượng ảnh và nhãn không khớp!")

images, labels = [], []
start_time = time.time()

print("Loading images and labels...")
for img_file, label_file in tqdm(zip(img_files, label_files), total=len(img_files)):
    img = io.imread(os.path.join(img_path_folder, img_file))
    label_img = io.imread(os.path.join(label_path_folder, label_file))
    
    img = transform.resize(img, (96, 256), anti_aliasing=True)
    label_img = transform.resize(label_img, (96, 256), anti_aliasing=True)
    
    if len(label_img.shape) == 3:
        label_img = color.rgb2gray(label_img)
        label_img = (label_img * 255).astype(np.uint8)
    
    images.append(img)
    labels.append(label_img)

# 🛠️ Hàm phát hiện cạnh Sobel + Canny
def extract_edges(image_gray):
    sobel_edges = filters.sobel(image_gray)  # Phát hiện cạnh bằng Sobel
    canny_edges = feature.canny(image_gray, sigma=1)  # Phát hiện cạnh bằng Canny
    return np.stack([sobel_edges, canny_edges], axis=-1)

# 🔥 Trích xuất đặc trưng nâng cao, thêm phát hiện cạnh
def extract_features(image):
    gray_img = color.rgb2gray(image)
    features_list = []
    
    # Multiscale Basic Features (chuyển về vector 1D)
    basic_features = feature.multiscale_basic_features(
        image, intensity=True, edges=True, texture=True,
        sigma_min=1, sigma_max=16, channel_axis=-1
    ).reshape(-1)  # Flatten về 1D
    features_list.append(basic_features)
    
    # HOG
    features_list.append(hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True))
    
    # DAISY
    features_list.append(daisy(gray_img, step=16, radius=8, rings=3, histograms=8, orientations=8).flatten())
    
    # Edge Features (Sobel + Canny)
    edges = extract_edges(gray_img).flatten()
    features_list.append(edges)

    return np.hstack(features_list)  # Ghép tất cả thành vector 1D

if os.path.exists(features_path):
    print("Loading pre-extracted features...")
    data = np.load(features_path)
    X_train, y_train = data["X"], data["y"]
else:
    print("Extracting enhanced features...")
    extracted_features = [extract_features(img) for img in tqdm(images, total=len(images))]
    X_train = np.vstack(extracted_features)
    y_train = np.concatenate([label.flatten() for label in labels]).astype(np.uint8)
    np.savez_compressed(features_path, X=X_train, y=y_train)

# Giảm số lượng mẫu tránh lỗi bộ nhớ
sample_size = min(1000000, X_train.shape[0])
sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train_sampled = X_train[sample_indices]
y_train_sampled = y_train[sample_indices]

# Train hoặc tải mô hình
if os.path.exists(model_path):
    print("Loading trained model...")
    clf = joblib.load(model_path)
else:
    print("Training model with enhanced features...")
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=20, max_samples=0.1, class_weight='balanced')
    clf.fit(X_train_sampled, y_train_sampled)
    joblib.dump(clf, model_path, compress=3)
    print("Model saved!")

end_time = time.time()
print(f"Process completed in {end_time - start_time:.2f} seconds.")
