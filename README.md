# Image Restoration in Frequency Domain

## Problem Statement

### Problem
Given a blurred image of a heart, the task is to recover the degraded function $H(u,v)$. It is known that, the bottom right corner crosshair image before degraded, is 3 pixels wide, 30 pixels long, and had an intensity of 255.

Further more, try to restore the original image.

The heart image is given as follow:

<div align="center">
    <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/heart.jpg" target="_blank">
        <img src="image/heart.jpg" alt="logo" style="width: 350px; height: auto; align: center">
    </a>
</div>

### Assumptions

Upon solving this problem, we consider the following assumptions:
- <b>Noise removed:</b> We consider the image is not corrupted by noise, e.g. no salt or pepper, hence absence of smoothing the image.
- <b>Gaussian or Butterworth:</b> We assume the blurred image is filtered using Gaussian or Butterworth lowpass filters, other techiques are not considered.

## Approach

TODO

## Experimental Results

### Gaussian Estimator

<div align="center">
  <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/result_gaussian.png" target="_blank">
    <img src="image/result_gaussian.png" alt="Description" width="450"/>
  </a>
  <p><em>Gaussian lowpass estimated results</em></p>
</div>


### Butterworth Estimator
<div align="center">
    <a href="https://github.com/thaiquangphat/Image-Restoration-in-Frequency-Domain/blob/main/image/result_butterworth.png" target="_blank">
        <img src="image/result_butterworth.png" alt="Description" width="450"/>
    </a>
  <p><em>Butterworth lowpass estimated results</em></p>
</div>

# Use The Project

Feel free to use the code, here are the instructions

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
python main.py image/heart.jpg <mode>
```

Where `mode` indicates the estimator, `mode = 1` for Gaussian estimator, and `mode = 2` for Butterworth filter.