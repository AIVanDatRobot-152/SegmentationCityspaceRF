import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, feature, color, transform
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import os
import joblib
from tqdm import tqdm

# Đường dẫn
img_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img"
label_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\label"
model_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\trained_model3.pkl"

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

# Load ảnh và nhãn
images, labels = [], []
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

# Hàm trích xuất đặc trưng
sigma_min, sigma_max = 1, 20
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True, edges=False, texture=True,
    sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1
)

# Dự đoán và hiển thị kết quả
def visualize_prediction(image_idx):
    img = images[image_idx]
    features = features_func(img)
    result = clf.predict(features.reshape(-1, features.shape[-1])).reshape(96, 256)
    
    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
    ax[0].set_title('Ảnh gốc & biên phân đoạn')
    ax[1].imshow(result)
    ax[1].set_title('Kết quả phân đoạn')
    ax[2].imshow(labels[image_idx])
    ax[2].set_title('Nhãn gốc')
    fig.tight_layout()
    plt.show()

# Chọn ảnh để xem kết quả
while True:
    try:
        idx = input(f"Nhập số thứ tự ảnh (0-{len(images)-1}) hoặc 'exit' để thoát: ")
        if idx.lower() == 'exit':
            break
        idx = int(idx)
        if 0 <= idx < len(images):
            visualize_prediction(idx)
        else:
            print("Số không hợp lệ, hãy nhập lại!")
    except ValueError:
        print("Hãy nhập một số hợp lệ!")
