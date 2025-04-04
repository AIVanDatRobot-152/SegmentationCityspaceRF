import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette('husl')

# Đường dẫn tới các thư mục ảnh và nhãn
img_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\img"
label_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\train\label"
val_img_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\img"
val_label_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\Citispaces_Segmentation_Random_Forest\DataCitiSpaces\val\label"

# Hàm để đếm số lượng ảnh trong từng lớp
def count_images_in_class(img_folder):
    class_counts = {}
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            # Lấy tên lớp từ tên thư mục chứa ảnh (giả sử thư mục con có tên lớp)
            class_name = os.path.basename(root)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

# Đếm số lượng ảnh trong tập train và val
train_img_class_counts = count_images_in_class(img_path)
train_label_class_counts = count_images_in_class(label_path)
val_img_class_counts = count_images_in_class(val_img_folder)
val_label_class_counts = count_images_in_class(val_label_folder)

# Tạo dataframe cho việc vẽ biểu đồ
train_img_class_counts_df = pd.DataFrame(list(train_img_class_counts.items()), columns=['Class', 'Train img Count'])
train_label_class_counts_df = pd.DataFrame(list(train_label_class_counts.items()), columns=['Class', 'Train label Count'])
val_img_class_counts_df = pd.DataFrame(list(val_img_class_counts.items()), columns=['Class', 'Validation img Count'])
val_label_class_counts_df = pd.DataFrame(list(val_label_class_counts.items()), columns=['Class', 'Validation label Count'])

# Kết hợp dữ liệu
class_distribution = pd.merge(train_img_class_counts_df, train_label_class_counts_df, on='Class', how='outer')
class_distribution = pd.merge(class_distribution, val_img_class_counts_df, on='Class', how='outer')
class_distribution = pd.merge(class_distribution, val_label_class_counts_df, on='Class', how='outer')

# Điền giá trị NaN bằng 0
class_distribution = class_distribution.fillna(0)

# Vẽ biểu đồ phân bố số lượng ảnh
plt.figure(figsize=(12, 6))

# Vẽ cột cho train và validation với màu sắc khác nhau
class_distribution.set_index('Class').plot(kind='bar', stacked=False, figsize=(12, 6), color=['royalblue', 'orange', 'green', 'red'])

plt.title('Số lượng ảnh phân bố cho từng lớp (Train và Validation)')
plt.xlabel('Lớp')
plt.ylabel('Số lượng ảnh')
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()
