import numpy as np
import cv2
import matplotlib.pyplot as plt

# Coordinates regions
regions = {
    "crosshair": [554, 597, 493, 535],
    "crack": [150, 315, 251, 321]
}

# Select the region want to crop out
crop_x_start, crop_x_end, crop_y_start, crop_y_end = regions["crack"]

# Define the crop crosshair
def cropImage(image):
    # Crop the region to be compared with the restored version
    cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return cropped_image

if __name__ == "__main__":
    input = cv2.imread("image/heart.jpg", cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread("image/restore.jpg", cv2.IMREAD_GRAYSCALE)
    bright = cv2.imread("image/brightened.jpg", cv2.IMREAD_GRAYSCALE)
    sharp = cv2.imread("image/sharpened.jpg", cv2.IMREAD_GRAYSCALE)

    input_cross = cropImage(input)
    restore_cross = cropImage(restored)
    bright_cross = cropImage(bright)
    sharpen_cross = cropImage(sharp)

    plt.figure(figsize=(8, 6))

    plt.suptitle("Cropped Cracks", fontsize=16, y=0.98)

    plt.subplot(2, 2, 1)
    plt.title("Input Blurred Image")
    plt.imshow(input_cross, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Restored Image")
    plt.imshow(restore_cross, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Brighten Restored Image")
    plt.imshow(bright_cross, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Enhanced Restored Image")
    plt.imshow(sharpen_cross, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()