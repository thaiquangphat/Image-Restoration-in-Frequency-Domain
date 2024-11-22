# Image Restoration in Frequency Domain

## Problem Statement

### Problem
Given a blurred image of a heart, the task is to recover the degraded function. It is known that, the bottom right corner crosshair image before degraded, is 3 pixels wide, 30 pixels long, and had an intensity of 255.

Further more, try to restore the original image.

The heart image is given as follow:

<div align="center">
    <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/heart.jpg" target="_blank">
        <img src="image/heart.jpg" alt="logo" style="width: 350px; height: auto; align: center">
    </a>
    <p><em>Blurred heart image</em></p>
</div>

### Assumptions

Upon solving this problem, we consider the following assumptions:
- <b>No knowledge of the original heart image:</b> We do not have the detailed heart image, thus at the restored result, we accept to our understanding.
- <b>Gaussian or Butterworth:</b> We assume the blurred image is filtered using either Gaussian or Butterworth lowpass filters, other methods are not considered.

## Approach

A detailed explanation of our approach is given in the <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/presentation.pdf" target="_blank">
presentation file</a>. This document contains a step-by-step instruction of how to obtain resulted images.

## Experimental Results

### Gaussian Estimator

As the document present, here is the result obtained using the Gaussian filter.
<div align="center">
  <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/result_gaussian.png" target="_blank">
    <img src="image/result_gaussian.png" alt="Description" width="450"/>
  </a>
  <p><em>Gaussian lowpass estimated results</em></p>
</div>


### Butterworth Estimator

For comparision, here is a result using the Butterworth filter, whose approach is similar to one made with the Gaussian filter.
<div align="center">
    <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/result_butterworth.png" target="_blank">
        <img src="image/result_butterworth.png" alt="Description" width="450"/>
    </a>
  <p><em>Butterworth lowpass estimated results</em></p>
</div>

# Use The Project

## File description

- `main.py`: Calling the restoration procedure from `restore.py`.
- `lowpass.py`: Compare the difference when applying ideal, Gaussian and Butterworth lowpass filter.
- `restore.py`: A detail restoration procedure.
- `ultis.py`: Includes some side functions, such as the Gaussian, Butterworth lowpass filters, ...
- `fourier.py`: Visualize the Fourier spectrum images.

## Run & Installation

Feel free to use the code, here are the instructions:

Clone the repository

```bash
git clone https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain.git
```

Change directory to the project
```bash
cd Image-Restoration-in-Frequency-Domain
```

To directly test the project, run
```bash
python main.py -i image/heart.jpg -m <mode> -s <save>
```
Arguments:
- `-i`: the image path.
- `-m`: indicates the estimator, `mode = 1` for Gaussian estimator, and `mode = 2` for Butterworth filter.
- `-s`: whether to save the result figue, default = `False`.