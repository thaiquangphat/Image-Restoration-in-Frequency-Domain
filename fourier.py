import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from ultis import gaussianKernel, butterworthKernel

crop_x_start, crop_x_end = 552, 598  # Adjust these coordinates based on the image
crop_y_start, crop_y_end = 492, 538  # Adjust these coordinates based on the image

# Define the crop crosshair
def cropCrossHair(blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return cropped_blurred_image

# Define the ideal synthetic crosshair
def idealCrossHair(cropped_blurred_image, blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    h, w = cropped_blurred_image.shape

    # Create the synthetic ideal crosshair for comparison
    ideal_crosshair = np.zeros_like(cropped_blurred_image, dtype=np.uint8)
    ideal_width, ideal_length = 3, 35  # Crosshair dimensions
    center_y, center_x = cropped_blurred_image.shape[0] // 2, cropped_blurred_image.shape[1] // 2

    # Create the horizontal line of the crosshair
    ideal_crosshair[center_y - ideal_width // 2: center_y + ideal_width // 2 + 1,
                    center_x - ideal_length // 2: center_x + ideal_length // 2 + 1] = 255

    # Create the vertical line of the crosshair
    ideal_crosshair[center_y - ideal_length // 2: center_y + ideal_length // 2 + 1,
                    center_x - ideal_width // 2: center_x + ideal_width // 2 + 1] = 255
    
    return ideal_crosshair

if __name__ == "__main__":
    blurred_image_input = cv2.imread("image/heart.jpg", cv2.IMREAD_GRAYSCALE)
    cropped_cross  = cropCrossHair(blurred_image_input)
    ideal_crosshair = idealCrossHair(cropped_cross, blurred_image_input)

    h, w = cropped_cross.shape
    lowpass = gaussianKernel(h, w, 7)

    G_fft = fft2(cropped_cross)
    F_fft = fft2(ideal_crosshair)

    G_shift = fftshift(G_fft)
    F_shift = fftshift(F_fft)
    F_shift = F_shift * lowpass
    G_F_shift = G_shift / F_shift

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.title("Fourier of Ideal")
    plt.imshow(np.abs(F_shift), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Fourier of Blurred")
    plt.imshow(np.abs(G_shift), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Fourier of G/F")
    plt.imshow(np.abs(G_F_shift), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Ideal Cross")
    plt.imshow(ideal_crosshair, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Cropped of Blurred")
    plt.imshow(cropped_cross, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    f_restore = np.abs(ifft2(ifftshift(F_shift)))

    f_restore = cv2.normalize(f_restore, None, 0, 255, cv2.NORM_MINMAX)
    f_restore = np.uint8(f_restore)

    plt.figure(figsize=(10, 6))

    plt.title("Blurred Cross")
    plt.imshow(f_restore, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    # plt.show()
