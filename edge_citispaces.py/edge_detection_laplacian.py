import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # Thêm thư viện này


def apply_laplacian_operator(gray_image):
    edge_image = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
    edge_image = cv2.convertScaleAbs(edge_image)  # Chuyển về dạng 0-255
    return edge_image

def main():
    image_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = apply_laplacian_operator(gray_image)

    output_path = "edge_detected_image.jpg"
    cv2.imwrite(output_path, edge_image)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.imshow(edge_image, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")
    plt.show() 

if __name__ == "__main__":
    main()
