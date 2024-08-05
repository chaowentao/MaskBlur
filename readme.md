This repository contains the implementation of the following paper:

  MaskBlur: Spatial and Angular Data Augmentation for Light Field Image Super-Resolution
# Get Started

## Environment Installation

```bash
conda env create -f environment.yaml
```

## Dataset Preparation

Following previous methods, we use five mainstream LF image datasets, i.e., EPFL, HCINew, HCIold, INRIA, and STFgantry for our LFSR experiments and convert the input LF image to YCbCr color space and only super-resolve the Y channel of the image, upsampling the Cb and Cr channel images bicubic. Please refer to https://github.com/ZhengyuLiang24/BasicLFSR


## Pretrained Model

We provide the pretrained InterNet, ATO model in the folder ``pth`` of this repository.


## Commands for Training and Test
* **Run **`train.py`** to perform network training. Example for training [InterNet] on 5x5 angular resolution for 4x SR:**
  ```
  ### InterNet ###
  python train.py --model_name LF_InterNet --angRes 5 --scale_factor 4 --batch_size 4 --augment maskblur --prob 0.25 --mask_ratio 0.5 --mask_patch 4 --drop_prob 0.75
  ```
* **Run **`test.py`** to perform network inference. Example for test [InterNet] on 5x5 angular resolution for 4xSR:**
  ```
  ### InterNet ###
  python test.py --model_name LF_InterNet --angRes 5 --scale_factor 4 --batch_size 4 --augment maskblur --prob 0.25 --mask_ratio 0.5 --mask_patch 4 --drop_prob 0.75 --use_pre_ckpt true --path_pre_pth pth/LF_InterNet_5x5_4x_model.pth
  ```
  
---
The code is modified and heavily borrowed from BasicLFSR: [https://github.com/ZhengyuLiang24/BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)

The code they provided is greatly appreciated.