import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, feature, future, color, transform
from sklearn.ensemble import RandomForestClassifier
from functools import partial

img_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png"
label_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\label\train1.png"

img = io.imread(img_path)
label_img = io.imread(label_path)

img = transform.resize(img, (96, 256), anti_aliasing=True)
label_img = transform.resize(label_img, (96, 256), anti_aliasing=True)

if len(label_img.shape) == 3:
    label_img = color.rgb2gray(label_img)
    label_img = (label_img * 50).astype(np.uint8)

training_labels = label_img

sigma_min, sigma_max = 1, 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True, edges=False, texture=True,
    sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1
)
features = features_func(img)

# Huấn luyện Random Forest
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)

# Hiển thị kết quả
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
ax[0].set_title('Ảnh gốc & biên phân đoạn')
ax[1].imshow(result)
ax[1].set_title('Kết quả phân đoạn')
ax[2].imshow(label_img)
ax[2].set_title('Nhãn gốc')
fig.tight_layout()
plt.show()
