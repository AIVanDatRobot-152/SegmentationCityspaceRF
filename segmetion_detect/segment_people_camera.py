import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo mô hình phân đoạn của Mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

### 📌 XỬ LÝ VIDEO (WEBCAM)
cap = cv2.VideoCapture(0)  # Mở webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang RGB để dùng Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segment.process(frame_rgb)
    mask = results.segmentation_mask
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Tạo hiệu ứng phân đoạn (tô màu xanh người)
    person_overlay = np.zeros_like(frame)
    person_overlay[:] = [0, 255, 0]  # Xanh lá cây

    segmented_frame = np.where(binary_mask[:, :, None] == 1, person_overlay, frame)

    # Hiển thị video trực tiếp
    cv2.imshow("Webcam - Phân đoạn người", segmented_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
