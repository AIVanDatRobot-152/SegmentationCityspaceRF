import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_canny(gray_image, sigma=0.5):

    median = np.median(gray_image)
    low_threshold = int(max(0, (1.0 - sigma) * median))
    high_threshold = int(min(255, (1.0 + sigma) * median))

    print(f"Ngưỡng tối ưu: Low={low_threshold}, High={high_threshold}")  
    edge_image = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    return edge_image

def overlay_edges_on_image(image, edge_image, edge_color=(55, 255, 255)):

    edge_mask = np.stack([edge_image] * 3, axis=-1)  
    highlighted_image = np.where(edge_mask > 0, edge_color, image)  

    return highlighted_image.astype(np.uint8)  # Chuyển đổi về kiểu dữ liệu hợp lệ

def main():
    image_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png"
    label_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\label\train1.png"
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = auto_canny(gray_image)

    # Chồng cạnh lên ảnh gốc
    highlighted_image = overlay_edges_on_image(image, edge_image)

    output_path = "edge_detected_overlay.jpg"
    cv2.imwrite(output_path, highlighted_image)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB))
    plt.title("Edge Detection")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(highlighted_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Edge Detection Overlay")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))
    plt.title("Label original")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
