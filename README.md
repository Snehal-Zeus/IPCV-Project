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

## 📂 Repository Structure

