import joblib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# ----- Thông tin cấu hình -----
model_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\random_forest_model_vjppro.pkl"
val_img_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\img"
val_label_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\label"
img_size = (96, 256)
test_img_name = "val17.png"

# ----- Mapping màu & lớp -----
color_to_class = {
    (143, 0, 1): 0,
    (13, 223, 222): 1,
    (34, 142, 106): 2,
    (127, 63, 128): 3,
    (70, 70, 70): 4,
    (180, 130, 70): 5,
    (230, 35, 240): 6,
    (150, 250, 150): 7
}

class_names = {
    0: "Car",
    1: "Sign",
    2: "Tree",
    3: "Road",
    4: "Building",
    5: "Sky",
    6: "Sidewalk",
    7: "Grass"
}

# ----- Hàm trích xuất đặc trưng cho 1 pixel -----
def extract_features(img, x, y, window_size=5):
    half = window_size // 2
    patch = img[max(0, y-half):y+half+1, max(0, x-half):x+half+1]

    r, g, b = img[y, x]
    hsv = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_RGB2HSV)[0, 0]

    height, width = img.shape[:2]
    norm_x, norm_y = x / width, y / height

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    else:
        lbp_hist = np.zeros(9)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2).mean()

    features = [
        r, g, b, hsv[0], hsv[1], hsv[2],
        norm_x, norm_y, *lbp_hist, grad_mag
    ]
    return np.array(features)

# ----- Hàm dự đoán segmentation 1 ảnh -----
def predict_image(model, img_path, label_path=None, img_size=(96, 256)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    h, w = img.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            feat = extract_features(img, x, y)
            pred_mask[y, x] = model.predict([feat])[0]

    # Hiển thị ảnh gốc và ảnh segmentation
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("🖼 Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=len(class_names)-1)
    plt.title("🔮 Predicted Segmentation")
    plt.colorbar(ticks=range(len(class_names)), label="Class ID")

    # Nếu có ground truth thì vẽ thêm
    if label_path and os.path.exists(label_path):
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, img_size)

        label_class = np.zeros((h, w), dtype=np.uint8)
        for color, class_id in color_to_class.items():
            mask = np.all(label == np.array(color), axis=-1)
            label_class[mask] = class_id

        plt.subplot(1, 3, 3)
        plt.imshow(label_class, cmap='jet', vmin=0, vmax=len(class_names)-1)
        plt.title("📌 Ground Truth")
        plt.colorbar(ticks=range(len(class_names)), label="Class ID")

    plt.suptitle(f"🧪 Prediction - {os.path.basename(img_path)}", fontsize=16)
    plt.tight_layout()
    plt.show()

# ----- Chạy thử -----
rf = joblib.load(model_path)
test_img_path = os.path.join(val_img_folder, test_img_name)
test_label_path = os.path.join(val_label_folder, test_img_name)

predict_image(rf, test_img_path, test_label_path, img_size=img_size)
