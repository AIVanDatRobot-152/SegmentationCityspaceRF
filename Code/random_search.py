import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

# Đường dẫn và tham số
img_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\img"
label_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\label"
img_size = (48, 128)  # Giảm kích thước để tăng tốc độ

def mean_iou(y_true, y_pred):
    """Tính IoU cho multi-class segmentation"""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    present_classes = np.unique(y_true)
    ious = []
    
    for cls in present_classes:
        pred_inds = y_pred == cls
        target_inds = y_true == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0

iou_scorer = make_scorer(mean_iou, greater_is_better=True)

def load_data(img_path, label_path, max_samples=10):
    images = []
    labels = []
    
    try:
        # Kiểm tra đường dẫn
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path not found: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label path not found: {label_path}")

        # Lấy danh sách file (bỏ qua thư mục con nếu có)
        img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))][:max_samples]
        label_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))][:max_samples]

        if not img_files:
            raise ValueError("No image files found in directory")
        if not label_files:
            raise ValueError("No label files found in directory")

        for img_file, label_file in zip(img_files, label_files):
            # Đọc ảnh
            img = cv2.imread(os.path.join(img_path, img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
                
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Đọc nhãn
            label = cv2.imread(os.path.join(label_path, label_file), cv2.IMREAD_GRAYSCALE)
            if label is None:
                print(f"Warning: Could not read label {label_file}")
                continue
                
            label = cv2.resize(label, img_size, interpolation=cv2.INTER_NEAREST)
            
            images.append(img)
            labels.append(label)

        if len(images) == 0:
            raise ValueError("No valid images/labels were loaded")

        return np.array(images), np.array(labels)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def preprocess_data(images, labels):
    if images is None or len(images) == 0:
        raise ValueError("No images to preprocess")
    
    X = images.astype('float32') / 255.0
    X = X.reshape(len(images), -1)  # Flatten
    
    y = labels
    return X, y

if __name__ == "__main__":
    print("Loading data...")
    images, labels = load_data(img_path, label_path, max_samples=10)
    
    if images is None:
        print("Failed to load data. Exiting...")
        exit()
    
    print(f"Successfully loaded {len(images)} images")
    
    try:
        X, y = preprocess_data(images, labels)
        print("Data shape:", X.shape, y.shape)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
        }
        
        print("Starting Random Search...")
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(n_jobs=-1, random_state=42),
            param_distributions=rf_params,
            n_iter=4,
            cv=2,
            scoring=iou_scorer,
            verbose=2
        )
        
        rf_search.fit(X_train, y_train.reshape(len(y_train), -1))
        
        print("\nBest parameters found:", rf_search.best_params_)
        print("Best IoU score:", rf_search.best_score_)
        
        best_model = rf_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_iou = mean_iou(y_test, y_pred.reshape(y_test.shape))
        print(f"Test IoU: {test_iou:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")