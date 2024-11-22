import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help="Path to your image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)

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

def padImage(image):
    M, N = image.shape
    padded_image = cv2.copyMakeBorder(image, 0, M, 0, N, cv2.BORDER_CONSTANT, value=0)

    return padded_image

def idealKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float32)
    centerX, centerY = M / 2, N / 2

    for u in range(M):
        for v in range(N):
            dist = np.sqrt((u - centerX)**2 + (v - centerY)**2)
            if dist <= D0:
                H[u, v] = 1
    return H

def gaussianKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float_)
    centerX, centerY = M/2, N/2

    for u in range(M):
        for v in range(N):
            dist = (u - centerX)**2 + (v - centerY)**2
            H[u, v] = np.exp(-dist/(2*D0**2))

    return H

def butterworthKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float_)
    centerX, centerY, n = M/2, N/2, 2.25

    for u in range(M):
        for v in range(N):
            dist = np.sqrt((u - centerX)**2 + (v - centerY)**2)
            if dist == 0:
                H[u, v] = 1
            else:
                H[u, v] = 1 / (1 + (dist/D0)**(2*n))

    return H

def lowPass(image, filter, D0):
    M, N = image.shape

    imagePadded = padImage(image)

    P, Q = imagePadded.shape

    for u in range(P):
        for v in range(Q):
            imagePadded[u, v] *= (-1)**(u+v)

    imageFourier = np.fft.fft2(imagePadded)
    lowPassKernel = filter(P, Q, D0)

    G = imageFourier * lowPassKernel
    resImage = np.fft.ifft2(G).real

    for u in range(P):
        for v in range(Q):
            resImage[u, v] *= (-1)**(u+v)
    
    resImage = cv2.normalize(resImage, None, 0, 255, cv2.NORM_MINMAX)
    resImage = np.uint8(resImage)  # Convert to uint8 for displaying


    resImage = resImage[:M, :N]

    return resImage

croppedCrosshair = cropCrossHair(image)
crosshair = idealCrossHair(croppedCrosshair, image)

# Initial the cutoff frequency
D0_ideal = 12
D0_gaussian = 4.5
D0_butterworth = 5.6

lowPassIdealImage = lowPass(crosshair, idealKernel, D0_ideal)
lowPassGaussianImage = lowPass(crosshair, gaussianKernel, D0_gaussian)
lowPassButterworthImage = lowPass(crosshair, butterworthKernel, D0_butterworth)


# Plot the images in a single frame
plt.figure(figsize=(10, 8))

# Plot Ideal low-pass filtered image
plt.subplot(1, 3, 1)
plt.imshow(lowPassIdealImage, cmap='gray')
plt.title(f"Ideal Lowpass Filter, $D_0$ = {D0_ideal}")
plt.axis('off')

# Plot Gaussian low-pass filtered image
plt.subplot(1, 3, 2)
plt.imshow(lowPassGaussianImage, cmap='gray')
plt.title(f"Gaussian Lowpass Filter, $D_0$ = {D0_gaussian}")
plt.axis('off')

# Plot Butterworth low-pass filtered image
plt.subplot(1, 3, 3)
plt.imshow(lowPassButterworthImage, cmap='gray')
plt.title(f"Butterworth Lowpass Filter, $D_0$ = {D0_butterworth}")
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()

plt.savefig('image/filters_comparison.png')