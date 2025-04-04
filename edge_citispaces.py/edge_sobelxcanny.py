import cv2
import numpy as np
import matplotlib.pyplot as plt

def upscale_image(image, scale=2):
    """Tăng độ phân giải ảnh bằng nội suy tuyến tính"""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def enhance_contrast(image):
    """Tăng cường tương phản ảnh bằng CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Tăng độ sắc nét bằng Unsharp Mask"""
    gaussian_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, gaussian_blurred, -0.5, 0)
    return sharpened

def apply_sobel_operator(gray_image):
    """Áp dụng bộ lọc Sobel để phát hiện cạnh chi tiết hơn"""
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_image = cv2.convertScaleAbs(magnitude)
    return edge_image

def apply_scharr_operator(gray_image):
    """Sử dụng bộ lọc Scharr để phát hiện cạnh rõ hơn Sobel"""
    grad_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_image = cv2.convertScaleAbs(magnitude)
    return edge_image

def auto_canny(gray_image, sigma=0.33):
    """Canny với ngưỡng động"""
    median = np.median(gray_image)
    low_threshold = int(max(0, (1.0 - sigma) * median))
    high_threshold = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray_image, low_threshold, high_threshold)

def apply_laplacian_of_gaussian(gray_image):
    """Bộ lọc LoG để phát hiện cạnh chính xác hơn"""
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def combine_edges(*edges):
    """Kết hợp nhiều bộ lọc cạnh"""
    combined = np.zeros_like(edges[0])
    for edge in edges:
        combined = cv2.addWeighted(combined, 0.5, edge, 0.5, 0)
    return combined

def overlay_edges_on_image(image, edge_image, edge_color=(0, 255, 0), alpha=0.4):
    """Chồng viền cạnh lên ảnh gốc với hiệu ứng hòa trộn"""
    edge_colored = np.zeros_like(image)
    edge_colored[:, :] = edge_color
    edge_colored = cv2.bitwise_and(edge_colored, edge_colored, mask=edge_image)
    blended = cv2.addWeighted(image, 1 - alpha, edge_colored, alpha, 0)
    return blended

def main():
    image_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    image = upscale_image(image, scale=1.5)
    image = enhance_contrast(image)
    image = sharpen_image(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    sobel_edges = apply_sobel_operator(blurred_image)
    scharr_edges = apply_scharr_operator(blurred_image)
    canny_edges = auto_canny(blurred_image)
    log_edges = apply_laplacian_of_gaussian(blurred_image)
    combined_edges = combine_edges(sobel_edges, scharr_edges, canny_edges, log_edges)
    highlighted_image = overlay_edges_on_image(image, combined_edges)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Enhanced Image"); plt.axis("off")
    plt.subplot(2, 3, 2); plt.imshow(sobel_edges, cmap="gray"); plt.title("Sobel Edges"); plt.axis("off")
    plt.subplot(2, 3, 3); plt.imshow(scharr_edges, cmap="gray"); plt.title("Scharr Edges"); plt.axis("off")
    plt.subplot(2, 3, 4); plt.imshow(canny_edges, cmap="gray"); plt.title("Canny Edges"); plt.axis("off")
    plt.subplot(2, 3, 5); plt.imshow(log_edges, cmap="gray"); plt.title("LoG Edges"); plt.axis("off")
    plt.subplot(2, 3, 6); plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)); plt.title("Overlay Edges"); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
