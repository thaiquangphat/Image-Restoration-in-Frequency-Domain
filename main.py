import numpy as np
import cv2
import argparse
from restore import restoreWithGaussian, restoreWithButterworth
import matplotlib.pyplot as plt

def main(save=False):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image restoration using Gaussian or Butterworth filter.")
    parser.add_argument('-i', '--image', type=str, required=True, help="Path to the blurred input image.")
    parser.add_argument("-m", '--mode', type=int, choices=[1, 2], help="Restoration mode: 1 for Gaussian, 2 for Butterworth.")
    parser.add_argument("-s", '--save', type=bool, default=False, help="Restoration mode: 1 for Gaussian, 2 for Butterworth.")
    args = parser.parse_args()

    # Read the input blurred image
    blurred_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if blurred_image is None:
        print(f"Error: Unable to read image at {args.image}")
        return

    # Perform restoration based on the selected mode
    if args.mode == 1:
        restored_image, brighten_image = restoreWithGaussian(blurred_image, True)
    elif args.mode == 2:
        restored_image, brighten_image = restoreWithButterworth(blurred_image, True)

    # Post processing: enhance image using convolution
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    
    # Preprocess by smoothing the image
    enhanced_image = cv2.filter2D(restored_image, -1, kernel)
    
    # Save the side-by-side comparision image
    plt.figure(figsize=(8, 6))

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
    plt.title("Enhanced Restored Image")
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()

    # Save the result
    if save == True:
        if args.mode == 1:
            plt.savefig('image/result_gaussian.png')
        elif args.mode == 2:
            plt.savefig('image/result_butterworth.png')

    plt.show()

if __name__ == "__main__":
    main(save=True)
