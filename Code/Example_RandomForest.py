import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import datetime
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Định nghĩa đường dẫn dữ liệu
extract_path = "DataCitiSpaces.vs1.folder"
train_path = os.path.join(extract_path, "train")
valid_path = os.path.join(extract_path, "val")

# # Hàm trích xuất đặc trưng HOG từ ảnh
# def extract_hog_features(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (64, 64))  # Resize ảnh về 64x64
#     features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
#                       orientations=9, block_norm='L2-Hys', visualize=True)
#     return features

# # Hàm load dữ liệu từ thư mục
# def load_data_from_folder(folder_path):
#     X, y = [], []
#     for label in os.listdir(folder_path):
#         label_path = os.path.join(folder_path, label)
#         if os.path.isdir(label_path):
#             for image_name in os.listdir(label_path):
#                 image_path = os.path.join(label_path, image_name)
#                 X.append(extract_hog_features(image_path))
#                 y.append(label)
#     return np.array(X), np.array(y)

# Đánh giá mô hình
def evaluate_model(rf_model, X_valid, y_valid):
    y_pred = rf_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:\n", classification_report(y_valid, y_pred))
    
    return conf_matrix, report

# Hàm vẽ đồ thị confusion matrix và metrics
def plot_metrics_and_conf_matrix(cm, report, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")
    
    # Precision, Recall, F1-score Bar Chart
    class_labels = [str(label) for label in labels]
    precision = [report[label]['precision'] for label in class_labels]
    recall = [report[label]['recall'] for label in class_labels]
    f1_score = [report[label]['f1-score'] for label in class_labels]
    
    x = np.arange(len(class_labels))
    width = 0.2
    bars1 = axes[1].bar(x - width, precision, width=width, label='Precision')
    bars2 = axes[1].bar(x, recall, width=width, label='Recall')
    bars3 = axes[1].bar(x + width, f1_score, width=width, label='F1-score')
    
    # Hiển thị giá trị trên cột
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         '{:.2f}'.format(height), ha='center', fontsize=8)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_labels, rotation=30)
    axes[1].set_title("Metrics per Class")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig("random_forest_metric.png")
    plt.close()

# Thông số tối ưu của mô hình Random Forest
param_grid = {
    "n_estimators": np.int64(250),
    "min_samples_split": 5,
    "min_samples_leaf": 6,
    "max_features": 'log2',
    "max_depth": 30
}
# X_train, y_train = load_data_from_folder(train_path)
# X_valid, y_valid = load_data_from_folder(valid_path)

id_train = [
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz",
    "train_feature-emth-0-128x128-202503270255.npz"
]

id_val = [
    "val_feature-emth-0-128x128-202503270255.npz",
    "val_feature-emth-0-128x128-202503270255.npz",
    "val_feature-emth-0-128x128-202503270255.npz",
    "val_feature-emth-0-128x128-202503270255.npz",
    "val_feature-emth-0-128x128-202503270255.npz"
]

# Load dữ liệu từ các file .npz
train_data = []
val_data = []

for i in range(len(id_train)):
    data = np.load("train_feature_extracted/{}".format(id_train[i]))
    train_data.append(data)

for i in range(len(id_val)):
    data = np.load("val_feature_extracted/{}".format(id_val[i]))
    val_data.append(data)

# Gán dữ liệu vào biến X và y
X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Định nghĩa tham số cho Random Forest
param_grid = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2']
}

# Khởi tạo Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Tìm kiếm tham số tối ưu
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy'
    njob = -1,
    random_state=42,
)

print("\n Đang tìm kiếm tham số tối ưu...")
random_search.fit(X_train, y_train_encoded)
print("Tìm kiếm tham số hoàn tất.")
print(random_search.best_params_)

# Huấn luyện mô hình với tham số tốt nhất
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train_encoded)

# Đánh giá mô hình
accuracy = best_rf.score(X_val, y_val_encoded)
print(f"\n Độ chính xác trên tập kiểm tr: {accuracy:.4f}")
conf_matrix, report = evaluate_model(best_rf, X_val, y_val_encoded)

# Ghi lại thông tin vào file log
with open("train_history.log", "a") as log_file:
    log_file = open("train_history.log", "a")
    log_file.write("\n======================\n")
    log_file.write("Train size: {}\n".format(X_train.shape[0]))
    log_file.write("Validation size: {}\n".format(X_val.shape[0]))
    log_file.write("Feature extraction method: {}\n".format(
        "SIFT" if extract_method == 0 else "Manual library"))
    log_file.write("Best hyperparameter results: {}\n".format(random_search.best_params_))
    log_file.write("Accuracy score: {}\n".format(accuracy))
    log_file.write("Model saved at path: {}\n".format(model_path))
    log_file.close()

# In kết quả cuối cùng
print("Trực quan hóa kết quả:")
plot_metrics_and_conf_matrix(id_train[i],conf_matrix, report, labels= np.unique(y_train_encoded))
