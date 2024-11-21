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

def restoreWithGaussian(blurred_image):
    u, v = blurred_image.shape

    # Crop out the cross hair section
    cropped_blurred_image = cropCrossHair(blurred_image)
    h, w = cropped_blurred_image.shape

    # Create the ideal synthetic crosshair
    ideal_crosshair = idealCrossHair(cropped_blurred_image, blurred_image)

    # Apply Fourier Transform to both images
    F_blurred = fft2(cropped_blurred_image)
    F_ideal = fft2(ideal_crosshair)

    # Shift the zero frequency component to the center
    F_blurred_shifted = fftshift(F_blurred)
    F_ideal_shifted = fftshift(F_ideal)

    # Compute the blurring function H(u,v)
    epsilon = 1e-10
    H_uv = F_blurred_shifted / (F_ideal_shifted + epsilon)

    # Calculate D0
    h_center, w_center = h//2, w//2
    D0 = 0.0
    for i in range(h):
        for j in range(w):
            dist = (i - h_center)**2 + (j - w_center)**2
            huv = np.log(H_uv[i, j])
            # d0 = np.sqrt(np.abs(-dist / huv) / 2)
            d0 = (np.sqrt(np.abs(-dist / huv))) / 2

            D0 += d0

    # Taking the mean
    D0 = (D0 / (h * w)) / 1.6 # For gaussian
    # print(f"Estimated D0: {D0}")

    # Define the estimate gaussian kernel
    G_shift = gaussianKernel(h, w, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.0001 # Regularization parameter
    H_uv_abs2 = np.abs(G_shift)**2
    restored_fft = (F_blurred_shifted * G_shift.conj()) / (H_uv_abs2 + K)

    # Inverse FFT to restore the image cross
    restored_image = np.abs(ifft2(ifftshift(restored_fft)))

    # Update D0 for entire image
    D0 = D0 * 11.55

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

    # Brightening the image
    f_restore_brightened = np.copy(f_restore)
    f_restore_brightened[f_restore_brightened > 5] += 30

    # Ensure pixel values do not exceed 255
    f_restore_brightened = np.clip(f_restore_brightened, 0, 255)

    return f_restore, f_restore_brightened

def restoreWithButterworth(blurred_image):
    u, v = blurred_image.shape

    # Crop out the cross hair section
    cropped_blurred_image = cropCrossHair(blurred_image)
    h, w = cropped_blurred_image.shape

    # Create the ideal synthetic crosshair
    ideal_crosshair = idealCrossHair(cropped_blurred_image, blurred_image)

    # Apply Fourier Transform to both images
    F_blurred = fft2(cropped_blurred_image)
    F_ideal = fft2(ideal_crosshair)

    # Shift the zero frequency component to the center
    F_blurred_shifted = fftshift(F_blurred)
    F_ideal_shifted = fftshift(F_ideal)

    # Compute the blurring function H(u,v)
    epsilon = 1e-10
    H_uv = F_blurred_shifted / (F_ideal_shifted + epsilon)

    # Calculate D0
    h_center, w_center = h//2, w//2
    D0 = 0.0

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - h_center)**2 + (j - w_center)**2)
            huv = H_uv[i, j]

            d0 = (dist / ((1 / huv) - 1)**(1/2*2.25))

            D0 += np.abs(d0)

    # Taking the mean
    D0 = (D0 / (h * w)) * 2.75 # For butterworth
    # print(f"Estimated D0: {D0}")

    # Define the estimate butterworth kernel
    G_shift = butterworthKernel(h, w, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.0001 # Regularization parameter
    H_uv_abs2 = np.abs(G_shift)**2
    restored_fft = (F_blurred_shifted * G_shift.conj()) / (H_uv_abs2 + K)

    # Inverse FFT to restore the image cross
    restored_image = np.abs(ifft2(ifftshift(restored_fft)))

    # Update D0 for entire image
    D0 = D0 * 11.5

    # Restore the image
    F_original_fft = fft2(blurred_image)
    F_original_fftshift = fftshift(F_original_fft)

    G_full_shift = butterworthKernel(u, v, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.005 # Regularization parameter
    H_uv_abs2 = np.abs(G_full_shift)**2
    restore = (F_original_fftshift * G_full_shift.conj()) / (H_uv_abs2 + K)

    f_restore = np.abs(ifft2(ifftshift(restore)))

    f_restore = cv2.normalize(f_restore, None, 0, 255, cv2.NORM_MINMAX)
    f_restore = np.uint8(f_restore)  # Convert to uint8 for displaying

    # Brightening the image
    f_restore_brightened = np.copy(f_restore)
    f_restore_brightened[f_restore_brightened > 5] += 25

    # Ensure pixel values do not exceed 255
    f_restore_brightened = np.clip(f_restore_brightened, 0, 255)

    return f_restore, f_restore_brightened