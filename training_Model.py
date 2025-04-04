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
import joblib  # Thêm thư viện để lưu/load model
from datetime import datetime

# Thiết lập giao diện đồ họa
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette('husl')

# Đường dẫn dữ liệu và model
IMG_DIR = "path_to_cityscapes/images/train"
LABEL_DIR = "path_to_cityscapes/labels/train"
MODEL_PATH = "cityscapes_segmentation_model.joblib"
FEATURES_PATH = "preprocessed_features.npz"  # Để lưu features đã xử lý

# Tham số
IMG_SIZE = 256
NUM_SAMPLES = 100000  # Số pixel để huấn luyện

def save_model(model, accuracy=None):
    """Lưu model và metadata"""
    model_info = {
        'model': model,
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'img_size': IMG_SIZE,
        'num_samples': NUM_SAMPLES
    }
    joblib.dump(model_info, MODEL_PATH)
    print(f"Đã lưu model vào {MODEL_PATH}")

def load_model():
    """Load model đã lưu"""
    if os.path.exists(MODEL_PATH):
        model_info = joblib.load(MODEL_PATH)
        print(f"Đã load model được huấn luyện ngày {model_info['timestamp']}")
        print(f"Độ chính xác trước đó: {model_info['accuracy']*100:.2f}%")
        return model_info['model']
    return None

def extract_features(img_path, label_path):
    """Trích xuất đặc trưng từ ảnh và nhãn phân đoạn"""
    # [Giữ nguyên phần extract_features của bạn]
    return features, label

def load_or_preprocess_data():
    """Load dữ liệu đã xử lý hoặc xử lý mới"""
    if os.path.exists(FEATURES_PATH):
        print("Đang load dữ liệu đã xử lý...")
        data = np.load(FEATURES_PATH)
        return data['images'], data['labels']
    
    print("Đang xử lý dữ liệu mới...")
    images, labels = [], []
    
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith('.png')])
    
    assert len(img_files) == len(label_files) == 2975, "Số lượng ảnh và nhãn không khớp"

    for img_name, label_name in tqdm(zip(img_files, label_files), total=len(img_files)):
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, label_name)
        
        try:
            features, label = extract_features(img_path, label_path)
            images.append(features)
            labels.append(label.flatten())
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {str(e)}")

    images = np.array(images)
    labels = np.array(labels)
    
    # Lưu dữ liệu đã xử lý để dùng sau này
    np.savez_compressed(FEATURES_PATH, images=images, labels=labels)
    return images, labels

def prepare_training_data(images, labels):
    """Chuẩn bị dữ liệu huấn luyện"""
    all_pixels = np.concatenate([images, labels.reshape(-1, 1)], axis=1)
    sampled_pixels = all_pixels[np.random.choice(all_pixels.shape[0], NUM_SAMPLES, replace=False)]
    X = sampled_pixels[:, :-1]
    y = sampled_pixels[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    """Huấn luyện model mới"""
    print("\nĐang tối ưu siêu tham số...")
    param_grid = {'var_smoothing': np.logspace(-12, -2, 20)}
    
    grid_search = GridSearchCV(
        GaussianNB(),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_nb = grid_search.best_estimator_
    
    print(f"\nTham số tối ưu: {grid_search.best_params_}")
    print(f"Độ chính xác tốt nhất: {grid_search.best_score_:.4f}")
    
    return best_nb

def evaluate_model(model, X_test, y_test):
    """Đánh giá model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nĐộ chính xác trên tập test: {accuracy*100:.2f}%")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred))
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    top_classes = np.argsort(-np.bincount(y_test.astype(int)))[:10]
    sns.heatmap(cm[np.ix_(top_classes, top_classes)], annot=True, fmt='d', cmap='Blues')
    plt.title('Ma trận nhầm lẫn (10 lớp phổ biến nhất)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    return accuracy

def main():
    # Kiểm tra và load model nếu có
    model = load_model()
    
    if model is None:
        # Load hoặc xử lý dữ liệu
        images, labels = load_or_preprocess_data()
        
        # Chuẩn bị dữ liệu huấn luyện
        X_train, X_test, y_train, y_test = prepare_training_data(images, labels)
        
        # Huấn luyện model
        model = train_model(X_train, y_train)
        
        # Đánh giá model
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Lưu model
        save_model(model, accuracy)
    else:
        # Nếu chỉ muốn đánh giá lại
        # Có thể load dữ liệu và đánh giá ở đây
        pass
    
    # Hiển thị kết quả phân đoạn
    display_segmentation_results(
        IMG_DIR.replace('train', 'val'), 
        LABEL_DIR.replace('train', 'val'), 
        model
    )

if __name__ == "__main__":
    main()