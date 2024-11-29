import numpy as np
import cv2
import argparse
from restore import restoreWithGaussian
import matplotlib.pyplot as plt

def brightenImage(image, intensity, threshold):
    # Brightening the image
    brightened = np.copy(image)
    brightened[image > intensity] += intensity

    # Ensure pixel values do not exceed 255
    brightened = np.clip(brightened, 0, 255)

    return brightened

def sharpenImage(image, kernel):
    enhanced_image = cv2.filter2D(image, -1, kernel)

    return enhanced_image

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image restoration using Gaussian or Butterworth filter.")
    parser.add_argument('-i', '--image', type=str, default='image/heart.jpg', help="Path to the blurred input image.")
    args = parser.parse_args()

    # Read the input blurred image
    blurred_image_input = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if blurred_image is None:
        print(f"Error: Unable to read image at {args.image}")
        return

    # Perform restoration
    restored_image = restoreWithGaussian(blurred_image_input)

    # Post processing: brightening and sharpening image
    # Brightening
    brighten_image = brightenImage(restored_image, 20, 5)

    # Sharpening
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    enhanced_image = sharpenImage(restored_image, kernel)

    # Save the image individually
    cv2.imwrite("image/restore.jpg", restored_image)
    cv2.imwrite("image/brightened.jpg", brighten_image)
    cv2.imwrite("image/sharpened.jpg", enhanced_image)
    
    # Save the side-by-side comparision image
    plt.figure()

    plt.suptitle("Image Restoration and Enhancement Comparison", fontsize=16, y=0.98)

    plt.subplot(2, 2, 1)
    plt.title("Input Blurred Image")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Restored Image")
    plt.imshow(restored_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Brighten Restored Image")
    plt.imshow(brighten_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Sharpened Restored Image")
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()

    # Save the result
    plt.savefig('image/result_comparison.png')

    plt.show()

if __name__ == "__main__":
    main()
