import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # Thêm thư viện này


def apply_prewitt_operator(gray_image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Kernel theo X
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # Kernel theo Y

    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)

    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_image = cv2.convertScaleAbs(magnitude)

    return edge_image

def main():
    image_path = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img\train1.png"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = apply_prewitt_operator(gray_image)

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
