import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from ultis import gaussianKernel

# Coordinates regions
regions = {
    "crosshair": [554, 597, 493, 535],
    "crack": [150, 315, 251, 321]
}

# Select the region want to crop out
crop_x_start, crop_x_end, crop_y_start, crop_y_end = regions["crosshair"]

# Define the crop crosshair
def cropCrossHair(blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return cropped_blurred_image

# Define the ideal synthetic crosshair
def idealCrossHair(blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    h, w = cropped_blurred_image.shape

    # Create the synthetic ideal crosshair for comparison
    ideal_crosshair = np.zeros_like(cropped_blurred_image, dtype=np.uint8)
    ideal_width, ideal_length = 3, 33  # Crosshair dimensions 33 24
    center_y, center_x = cropped_blurred_image.shape[0] // 2, cropped_blurred_image.shape[1] // 2

    intensity = 240

    # Create the horizontal line of the crosshair
    ideal_crosshair[center_y - ideal_width // 2: center_y + ideal_width // 2 + 1,
                    center_x - ideal_length // 2: center_x + ideal_length // 2 + 1] = intensity

    # Create the vertical line of the crosshair
    ideal_crosshair[center_y - ideal_length // 2: center_y + ideal_length // 2 + 1,
                    center_x - ideal_width // 2: center_x + ideal_width // 2 + 1] = intensity
    
    return ideal_crosshair

def restoreWithGaussian(blurred_image):
    u, v = blurred_image.shape

    # Crop out the cross hair section
    cropped_blurred_image = cropCrossHair(blurred_image)
    h, w = cropped_blurred_image.shape

    # Create the ideal synthetic crosshair
    ideal_crosshair = idealCrossHair(blurred_image)

    # Apply Fourier Transform to both images
    F_blurred = fft2(cropped_blurred_image)
    F_ideal = fft2(ideal_crosshair)

    # Shift the zero frequency component to the center
    F_blurred_shifted = fftshift(F_blurred)
    F_ideal_shifted = fftshift(F_ideal)

    # Compute the blurring function H(u,v)
    K = 0.0001
    F_shift_abs2 = np.abs(F_ideal_shifted)**2
    H_uv = (F_blurred_shifted.astype(int) * F_ideal_shifted.conj().astype(int)) / (F_shift_abs2.astype(int) + K)

    # Calculate estimated D0
    h_center, w_center = h//2, w//2
    D0 = 0.0
    for i in range(h):
        for j in range(w):
            dist = (i - h_center)**2 + (j - w_center)**2
            Huv = 0.0

            # Handle exception cases
            Huv = 1.0001 if H_uv[i, j] <= 0 else H_uv[i, j]
            huv = np.log(Huv)
            t = 0 if -dist / (2*huv) <= 0 else -dist / (2*huv)
            
            d0 = np.sqrt(np.abs(t))

            D0 += d0

    # Taking the mean
    D0 = (D0 / (h * w))
    print(f"Estimated D0: {D0}")

    # Define the estimate gaussian kernel
    G_shift = gaussianKernel(h, w, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.0001 # Regularization parameter
    H_uv_abs2 = np.abs(G_shift)**2
    restored_fft = (F_blurred_shifted * G_shift.conj()) / (H_uv_abs2 + K)

    # Inverse FFT to restore the image cross
    restored_image = np.abs(ifft2(ifftshift(restored_fft)))

    plt.figure(figsize=(9,3))
    plt.suptitle("Ideal Crosshair and Its Restoration")

    plt.subplot(1, 3, 1)
    plt.title("Ideal Crosshair")
    plt.imshow(ideal_crosshair, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Blurred Crosshair")
    plt.imshow(cropped_blurred_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Restored Crosshair")
    plt.imshow(restored_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('image/restore_crosshair.jpg')

    # Update D0 for entire image
    D0 = D0 * (u / w)
    print(f"Scaled coefficient k: {u/h}")

    # Restore the image
    F_original_fft = fft2(blurred_image)
    F_original_fftshift = fftshift(F_original_fft)

    G_full_shift = gaussianKernel(u, v, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.005 # Regularization parameter
    H_uv_abs2 = np.abs(G_full_shift)**2
    restore = (F_original_fftshift * G_full_shift.conj()) / (H_uv_abs2 + K)

    f_restore = np.abs(ifft2(ifftshift(restore)))

    f_restore = cv2.normalize(f_restore, None, 0, 255, cv2.NORM_MINMAX)
    f_restore = np.uint8(f_restore)  # Convert to uint8 for displaying

    return f_restore