# IPCV-Project
This repository contains the IPCV Project.

# Adaptive Hologram Compression Using Adaptive DCT and Huffman Coding

This repository contains a complete implementation of a hologram-inspired image compression pipeline based on adaptive block-based DCT quantization and Huffman entropy coding. The method simulates hologram formation, isolates the first-order diffraction lobe via FFT filtering, applies adaptive JPEG-like compression, and evaluates image reconstruction using standard quality metrics.

---

## 📌 Overview

This project aims to efficiently compress hologram-like images while preserving critical visual features needed for reconstruction. The pipeline includes:

- Hologram simulation using cosine carrier frequencies  
- FFT-based off-axis spectral analysis  
- First-order lobe extraction using circular frequency masking  
- Adaptive DCT-based compression using dynamic quantization thresholds  
- Huffman entropy coding for further file-size reduction  
- Reconstruction of RGB images  
- Quality analysis using MSE, PSNR, SSIM  
- File-size comparison before and after entropy compression  

---

## 🚀 Features

### **1. Hologram Simulation**
- Converts input RGB image to YCbCr  
- Simulates off-axis hologram using carrier frequencies  
- FFT applied to visualize spectral lobe structure  

### **2. Spectral Filtering**
- Circular mask isolates the first diffraction order  
- Unwanted DC and conjugate components removed  
- Inverse FFT gives a clean preprocessed intensity image  

### **3. Adaptive DCT Compression**
- Image divided into 8×8 blocks  
- DCT applied to each block  
- Quantization threshold chosen using pixel proportion rule:  
  - **Tmin** for high-detail blocks  
  - **Tmax** for smoother blocks  
- Functions similarly to JPEG but with dynamic quantization  

### **4. Huffman Entropy Coding**
- Quantized coefficients flattened  
- Frequency table built  
- Huffman tree generated  
- Encoded bitstream stored in a pickle file  

### **5. Reconstruction & Evaluation**
- Reconstructed Y, Cb, Cr merged into RGB  
- Global brightness normalization applied  
- Computes:  
  - **MSE**  
  - **PSNR**  
  - **SSIM**  
- Reports storage comparison across stages  

---
###References

[1] R. C. Gonzalez and R. E. Woods, Digital Image Processing, Pearson, 2018.
[2] G. K. Wallace, “The JPEG Still Picture Compression Standard,” Communications of the ACM, 1992.
[3] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” 2004.
[4] J. W. Goodman, Introduction to Fourier Optics, 3rd Edition, Roberts & Company, 2005.
[5]  An improved JPEG compression method for digital hologram based on adaptive quantization table. Lei Hu, Jinbin Gui, Xianfei Hu, Zhuojian Tong
      https://doi.org/10.1016/j.optcom.2025.131987
      

