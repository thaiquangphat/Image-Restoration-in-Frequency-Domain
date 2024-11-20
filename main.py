import cv2
import argparse
from restore import restoreWithGaussian, restoreWithButterworth

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image restoration using Gaussian or Butterworth filter.")
    parser.add_argument("image_path", type=str, help="Path to the blurred input image.")
    parser.add_argument("mode", type=int, choices=[1, 2], help="Restoration mode: 1 for Gaussian, 2 for Butterworth.")
    args = parser.parse_args()

    # Read the input blurred image
    blurred_image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if blurred_image is None:
        print(f"Error: Unable to read image at {args.image_path}")
        return

    # Perform restoration based on the selected mode
    if args.mode == 1:
        restored_image = restoreWithGaussian(blurred_image, True)
    elif args.mode == 2:
        restored_image = restoreWithButterworth(blurred_image, True)

    print("Image restored.")

if __name__ == "__main__":
    main()
