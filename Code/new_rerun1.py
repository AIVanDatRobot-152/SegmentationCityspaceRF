import joblib
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            accuracy_score, cohen_kappa_score)
from skimage.feature import local_binary_pattern

# Đường dẫn dữ liệu và model
val_img_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\img"
val_label_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\label"
model_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\random_forest_model_vjppro.pkl"
img_size = (96, 256)
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Bảng màu cho các lớp đối tượng
class_colors = {
    0: [143, 0, 1],      # Xe hơi
    1: [13, 223, 222],   # Biển báo
    2: [34, 142, 106],   # Cây cối
    3: [127, 63, 128],   # Đường
    4: [70, 70, 70],     # Tòa nhà
    5: [180, 130, 70],   # Bầu trời
    6: [230, 35, 240],   # Vỉa hè
    7: [150, 250, 150]   # Cỏ
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
    """Trích xuất 18 đặc trưng từ ảnh"""
    half = window_size // 2
    patch = img[max(0, y-half):y+half+1, max(0, x-half):x+half+1]
    
    # 1. Màu sắc RGB (3)
    r, g, b = img[y, x]
    
    # 2. Không gian màu HSV (3)
    hsv = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_RGB2HSV)[0,0]
    
    # 3. Vị trí pixel (2)
    height, width = img.shape[:2]
    norm_x, norm_y = x / width, y / height
    
    # 4. Đặc trưng texture LBP (9)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    else:
        lbp_hist = np.zeros(9)
    
    # 5. Độ lớn gradient (1)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # Tổng cộng 18 đặc trưng
    features = [
        r, g, b,                 # RGB
        hsv[0], hsv[1], hsv[2],  # HSV
        norm_x, norm_y,          # Vị trí
        *lbp_hist,               # LBP (9 bins)
        grad_mag                 # Gradient
    ]
    
    return np.array(features)

def process_image(img_path):
    """Tiền xử lý ảnh đầu vào"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    
    # Trích xuất đặc trưng cho từng pixel
    h, w = img.shape[:2]
    features = []
    for y in range(h):
        for x in range(w):
            features.append(extract_features(img, x, y))
    
    return np.array(features), img

def evaluate_model():
    """Đánh giá mô hình trên tập validation"""
    print("Đang tải mô hình...")
    model = joblib.load(model_path)
    
    all_preds = []
    all_labels = []
    image_fnames = os.listdir(val_img_folder)[:5]  # Thử với 5 ảnh đầu
    
    for fname in tqdm(image_fnames, desc="Đang đánh giá"):
        img_path = os.path.join(val_img_folder, fname)
        label_path = os.path.join(val_label_folder, fname)
        
        # Trích xuất đặc trưng
        features, img = process_image(img_path)
        
        # Dự đoán theo từng batch để tiết kiệm bộ nhớ
        pred_mask = np.zeros((img_size[0], img_size[1]))
        batch_size = 10000
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            preds = model.predict(batch)
            
            # Gán kết quả dự đoán vào mask
            for j, (y, x) in enumerate([(idx//img_size[1], idx%img_size[1]) 
                                    for idx in range(i, min(i+batch_size, len(features)))]):
                pred_mask[y,x] = preds[j]

        
        # Xử lý nhãn gốc
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (img_size[1], img_size[0]))
        label_mask = np.zeros_like(pred_mask)
        
        # Chuyển đổi màu RGB sang class index
        for class_id, color in class_colors.items():
            mask = np.all(label == np.array(color), axis=-1)
            label_mask[mask] = class_id
        
        # Lưu kết quả
        all_preds.extend(pred_mask.flatten())
        all_labels.extend(label_mask.flatten())
        
        # Hiển thị kết quả
        visualize_results(img, pred_mask, label_mask, fname)
    
    # Tính toán các chỉ số đánh giá
    calculate_metrics(all_labels, all_preds)

def visualize_results(img, pred_mask, true_mask, fname):
    """Hiển thị ảnh gốc, nhãn thực và kết quả dự đoán"""
    # Tạo mask màu
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    true_color = np.zeros_like(pred_color)
    
    for class_id, color in class_colors.items():
        pred_color[pred_mask == class_id] = color
        true_color[true_mask == class_id] = color
    
    # Vẽ kết quả
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_color)
    plt.title("Nhãn thực")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_color)
    plt.title("Dự đoán")
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f"result_{fname}.png"))
    plt.close()

def calculate_metrics(true, pred):
    """Tính toán và hiển thị các chỉ số đánh giá"""
    print("\n" + "="*50)
    print("CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH")
    print("="*50)
    
    # Lọc chỉ các lớp có trong true và pred (lớp có trong nhãn gốc và dự đoán)
    unique_labels = np.unique(np.concatenate([true, pred]))

    # Báo cáo phân loại
    print("\nBáo cáo phân loại:")
    print(classification_report(true, pred, target_names=[class_names[i] for i in unique_labels], labels=unique_labels))
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_names[i] for i in unique_labels], 
                yticklabels=[class_names[i] for i in unique_labels])
    plt.title("Ma trận nhầm lẫn")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Các chỉ số chính
    metrics = {
        'Độ chính xác': accuracy_score(true, pred),
        'Precision (trung bình)': precision_score(true, pred, average='macro', zero_division=0),
        'Recall (trung bình)': recall_score(true, pred, average='macro', zero_division=0),
        'F1-score (trung bình)': f1_score(true, pred, average='macro', zero_division=0),
        'Cohen Kappa': cohen_kappa_score(true, pred)
    }
    
    # Lưu kết quả với mã hóa UTF-8
    with open(os.path.join(output_dir, "metrics.txt"), 'w', encoding='utf-8') as f:
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
            f.write(f"{name}: {value:.4f}\n")
    
    # Biểu đồ các chỉ số
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Hiệu suất mô hình")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_chart.png"))
    plt.close()


if __name__ == "__main__":
    # Đánh giá mô hình
    evaluate_model()
    print(f"Đánh giá hoàn tất. Kết quả lưu tại {output_dir}")