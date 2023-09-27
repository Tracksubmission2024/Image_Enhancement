# Image_Enhancement

### Enhancing Image Contrast: A Novel Approach using Modified Transfer Function and Energy Curve Equalization

## Overview
<img src="Block Diagram.png" width="1000"/>

**NOTE:**
- The block diagram of proposed method is shown in above diagram.
- **Proposed_Method.m** is MATLAB code for our algorithm.
- **Image1.jpg** is sample image taken from our dataset as input.

## Datasets
- We have taken three datasets for testing our proposed algorithm.
- Three Datasets : 200 images from our dataset of athletes, 100 Images from Berkeley BSDS dataset, 30 images from the CEED2016 dataset.
- Below are the samples of images from our dataset **IIT Patna Sports-2023** .
<img src="Our Dataset Samples.JPG" width="1000"/>

**Samples of Our Dataset IIT Patna Sports-2023**
| Datasets | No. of images taken |
|-------|:---------:|
|IIT Patna Sports-2023 (Ours) |   200    |
| Berkeley BSDS |   100    | 
| CEED2016 |   30    |

## Results
**Qualitative Analysis**
- Visualization of input and enhanced output images from IIT Patna Sports-2023, BSDS, and CEED2016 datasets are shown below.
- Column from top to bottom specifies the low contrast input image, images with JHE, MMSICHE, FCCE, ECE with TF and proposed ECE with modified TF (ours).
<img src="Qualitative Results.png" width="600"/>

**Quantitative Analysis**

**For IIT Patna Sports-2023 Dataset (200 Images)**
| Method | PSNR(dB) | SSIM | EME |
|-------|:---------:|:---------:|:---------:|
| MMSICHE|   19.761    | 0.8496 | 4.9697 | 
| JHE |   23.042    | 0.7841 | 5.8608 |
| FCCE|   22.011   | 0.6865 | 8.8227 |
| TF using ECE |   23.206   | 0.8640 | 4.6823 | 
| Modified TF using ECE (Ours) |   25.028    | 0.9041 | 4.2124 |

**For BSDS Dataset (100 Imaages)**
| Method | PSNR(dB) | SSIM | EME |
|-------|:---------:|:---------:|:---------:|
| MMSICHE|   21.713    | 0.9063 | 5.2995 | 
| JHE |   22.235    | 0.7261 | 5.7808 |
| FCCE|   23.121    | 0.6225 | 9.1290 |
| TF using ECE |   25.212    | 0.9297 | 4.7515 | 
| Modified TF using ECE (Ours) |   27.129    | 0.9312 | 4.2126 |

**For CEED2016 Dataset (30 Images)**
| Method | PSNR(dB) | SSIM | EME |
|-------|:---------:|:---------:|:---------:|
| MMSICHE|   19.993    | 0.9072 | 4.4935 | 
| JHE |   21.821  | 0.7530 | 4.8880 |
| FCCE|   22.112   | 0.6167 | 7.9884 |
| TF using ECE |   23.012    | 0.9340 | 4.3021 | 
| Modified TF using ECE (Ours) |   25.929    | 0.9496 | 4.0108 |


**Results produced by** Input image **Image1.jpg** in the block diagram.

<img src="PSNR_vs_Alpha.jpg" width="400"/> <img src="SSIM_vs_Alpha.jpg" width="400"/> <img src="EME vs Alpha.jpg" width="400"/>


## Environment
- MATLAB R2023a **(Recommend **NOT** to use very old versions of MATLAB.)**

