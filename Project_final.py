# newproj3_with_entropy.py
# Based on user's original newproj3.py with minimal additions:
#  - save pre-entropy compressed data
#  - perform Huffman entropy coding on quantized DCT coefficients (after adaptive quantization)
#  - compute metrics (MSE, PSNR, SSIM on Y channel)
#  - show file sizes and compression ratios


import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os
import pickle
import heapq
from scipy import ndimage


# Helper plotting function (unchanged)

def plot_on_axis(ax, image, title, cmap='gray', vmin=0, vmax=255):
    if isinstance(image, Image.Image):
        plot_image = np.array(image)
    else:
        plot_image = image
    if plot_image.ndim == 3 and plot_image.shape[2] == 3:
        ax.imshow(plot_image.astype(np.uint8))
    else:
        ax.imshow(plot_image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


#preprocessing adn compression

def add_hologram_noise_and_preprocess(image_channel_pil):
    img = np.array(image_channel_pil, dtype=float)
    H, W = img.shape
    U_FREQ, V_FREQ = 0.15, 0.1
    norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    x, y = np.arange(W), np.arange(H)
    xx, yy = np.meshgrid(x, y)
    carrier = np.cos(2 * np.pi * (U_FREQ * xx + V_FREQ * yy))
    holo = 128 + 100 * norm * carrier

    f = np.fft.fftshift(np.fft.fft2(holo))
    spec_orig = 20 * np.log(np.abs(f) + 1e-6)
    LOBE_RADIUS = 35
    u0 = int(H//2 - V_FREQ*H)
    v0 = int(W//2 - U_FREQ*W)
    mask = np.sqrt((yy - u0)**2 + (xx - v0)**2) < LOBE_RADIUS
    f_filt = f * mask
    spec_filt = 20 * np.log(np.abs(f_filt) + 1e-6)
    img_pre = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filt)))
    img_pre = 255*(img_pre - img_pre.min())/(img_pre.max()-img_pre.min()+1e-9)
    return (Image.fromarray(np.uint8(np.clip(holo,0,255))),
            Image.fromarray(np.uint8(img_pre)),
            spec_orig, spec_filt)

def block_dct(b):  return dct(dct(b.T, norm='ortho').T, norm='ortho')
def block_idct(b): return idct(idct(b.T, norm='ortho').T, norm='ortho')

def apply_pdf_compression(image_channel_pil):
    img = np.array(image_channel_pil, dtype=float)
    H, W = img.shape
    I, Tmin, Tmax = 0.25, 12, 120
    recon = np.zeros_like(img)
    qmap = np.zeros_like(img)
    thr = max(np.mean(img), 1.0)

    for r in range(0, H, 8):
        for c in range(0, W, 8):
            block = img[r:r+8, c:c+8] - 128
            coeff = block_dct(block)
            P = np.sum(block > thr)/64.0
            T = Tmin if P > I else Tmax
            qmap[r:r+8, c:c+8] = 255 if P > I else 0
            q = np.round(coeff/(T))
            deq = q*T
            recon[r:r+8, c:c+8] = block_idct(deq)+128
    recon = np.clip(recon,0,255)
    return Image.fromarray(recon.astype(np.uint8)), Image.fromarray(qmap.astype(np.uint8))



# quantization steps but returns quantized coeffs (per-block)

def extract_quantized_coeffs(image_channel_pil, I=0.25, Tmin=12, Tmax=120):
    """Return list of quantized DCT coefficients (in zigzag-like flattened 8x8 block order)"""
    img = np.array(image_channel_pil, dtype=float)
    H, W = img.shape
    coeffs_list = []
    T_list = []
    thr = max(np.mean(img), 1.0)

    for r in range(0, H, 8):
        for c in range(0, W, 8):
            block = img[r:r+8, c:c+8] - 128
            coeff = block_dct(block)
            P = np.sum(block > thr)/64.0
            T = Tmin if P > I else Tmax
            T_list.append(T)
            q = np.round(coeff/(T)).astype(int)   # quantized coefficients as integers
            # flatten block in natural row-major order
            coeffs_list.append(q.flatten().tolist())
    # flatten list of blocks to a single list
    flat = [int(item) for sub in coeffs_list for item in sub]
    return flat, T_list


# Huffman coding implementation 

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = [HuffmanNode(f, s) for s,f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = HuffmanNode(a.freq + b.freq, None, a, b)
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode_list(data_list):
    # data_list: list of integers (quantized coeffs)
    freq = {}
    for v in data_list:
        freq[v] = freq.get(v,0) + 1
    tree = build_huffman_tree(freq)
    codebook = build_codes(tree, "")
    # encode into bitstring (string of '0'/'1') 
    encoded = "".join(codebook[v] for v in data_list)
    return encoded, codebook


# === ADDED ===
# Metrics: MSE, PSNR, SSIM (single-scale, Y-channel)

def mse(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.mean((a - b)**2)

def psnr(a, b, maxval=255.0):
    m = mse(a, b)
    if m == 0:
        return float('inf')
    return 10 * np.log10((maxval**2) / m)

def ssim_index(img1, img2, K1=0.01, K2=0.03, win_sigma=1.5):
    # Single-scale SSIM implementation (grayscale images)
    # Returns SSIM index (scalar)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # Gaussian window (11x11) approximation
    window_size = 11
    window = np.outer(
        np.exp(-((np.arange(window_size)-window_size//2)**2)/(2*win_sigma**2)),
        np.exp(-((np.arange(window_size)-window_size//2)**2)/(2*win_sigma**2))
    )
    window = window / np.sum(window)

    mu1 = ndimage.convolve(img1, window, mode='reflect')
    mu2 = ndimage.convolve(img2, window, mode='reflect')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndimage.convolve(img1*img1, window, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(img2*img2, window, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(img1*img2, window, mode='reflect') - mu1_mu2

    C1 = (K1*255)**2
    C2 = (K2*255)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


# Main:
try:
    input_path  = r"E:\DIPLAB\DIP-Project\new1\Final\Head (2).tiff" #image filename 
    output_path = "reconstructed_head_1.jpg"

    img_orig = Image.open(input_path).convert("RGB")
    W,H = img_orig.size
    W8, H8 = W - (W%8), H - (H%8)
    img_orig = img_orig.crop((0,0,W8,H8))

    # Convert to YCbCr
    img_ycbcr = img_orig.convert("YCbCr")
    Y, Cb, Cr = img_ycbcr.split()

    print("Simulating holograms...")
    Y_sim, Y_pre, Y_spec, Y_filt = add_hologram_noise_and_preprocess(Y)
    Cb_sim, Cb_pre, _, _ = add_hologram_noise_and_preprocess(Cb)
    Cr_sim, Cr_pre, _, _ = add_hologram_noise_and_preprocess(Cr)

    print("Applying adaptive JPEG-like compression...")
    Y_recon, Y_q = apply_pdf_compression(Y_pre)
    Cb_recon, Cb_q = apply_pdf_compression(Cb_pre)
    Cr_recon, Cr_q = apply_pdf_compression(Cr_pre)

    print("Merging and restoring color...")
    # Merge reconstructed YCbCr → RGB
    img_recon_ycbcr = Image.merge("YCbCr", (Y_recon, Cb_recon, Cr_recon))
    img_recon_rgb = img_recon_ycbcr.convert("RGB")

    arr_recon = np.array(img_recon_rgb, dtype=float)
    arr_orig  = np.array(img_orig, dtype=float)
    mean_ratio = np.mean(arr_orig)/np.mean(arr_recon)
    arr_recon = np.clip(arr_recon*mean_ratio,0,255).astype(np.uint8)
    img_recon_rgb = Image.fromarray(arr_recon)

    img_recon_rgb.save(output_path,"JPEG",quality=95)
    print(f"Final color JPEG saved: {output_path}")

    #Plotting Intermediate Results
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    ax=axes.ravel()
    plot_on_axis(ax[0],img_orig,'1. Original Color Input')
    plot_on_axis(ax[1],Y_sim,'2. Simulated Hologram (Y-Ch)')
    plot_on_axis(ax[2],Y_spec,'3. DFT Spectrum (Y-Ch)')
    plot_on_axis(ax[3],Y_filt,'4. Filtered Spectrum (Y-Ch)')
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    ax=axes.ravel()
    plot_on_axis(ax[0],Y_pre,'5. Pre-processed (Y-Ch)')
    plot_on_axis(ax[1],Y_q,'6. Adaptive Q-Map (Y-Ch)')
    plot_on_axis(ax[2],Y_recon,'7. Reconstructed (Y-Ch)')
    plot_on_axis(ax[3],img_recon_rgb,'8. Final Reconstructed (Color)')
    
    fig.suptitle('Intermediate Steps of PDF Hologram Compression',fontsize=16)

    plt.show()


    # Extract quantized coefficients 
    print("\nExtracting quantized DCT coefficients for entropy coding (per channel)...")
    Y_qcoeffs, Y_Ts = extract_quantized_coeffs(Y_pre)
    Cb_qcoeffs, Cb_Ts = extract_quantized_coeffs(Cb_pre)
    Cr_qcoeffs, Cr_Ts = extract_quantized_coeffs(Cr_pre)

  
    all_qcoeffs = Y_qcoeffs + Cb_qcoeffs + Cr_qcoeffs

    pre_entropy_filename = "pre_entropy_quantized_coeffs.pkl"
    with open(pre_entropy_filename, "wb") as f:
        pickle.dump({
            "Y_qcoeffs_len": len(Y_qcoeffs),
            "Cb_qcoeffs_len": len(Cb_qcoeffs),
            "Cr_qcoeffs_len": len(Cr_qcoeffs),
            "Y_Ts": Y_Ts,
            "Cb_Ts": Cb_Ts,
            "Cr_Ts": Cr_Ts,
            "all_qcoeffs_sample": all_qcoeffs[:1000], 
        }, f)
   
    with open("pre_entropy_all_qcoeffs_full.pkl", "wb") as f:
        pickle.dump(all_qcoeffs, f)

    size_pre_entropy = os.path.getsize("pre_entropy_all_qcoeffs_full.pkl")
    print(f"Saved pre-entropy quantized coefficients file: pre_entropy_all_qcoeffs_full.pkl ({size_pre_entropy} bytes)")

  
    print("Applying Huffman entropy coding to quantized coefficients...")
    # build frequency map
    freq = {}
    for v in all_qcoeffs:
        freq[v] = freq.get(v, 0) + 1
    # build tree & codebook
    htree = build_huffman_tree(freq)
    codebook = build_codes(htree, "")
    # encode bitstring
    encoded_bitstring = "".join(codebook[v] for v in all_qcoeffs)

    # Save final entropy-coded result (pickle)
    entropy_filename = "final_entropy_compressed.pkl"
    with open(entropy_filename, "wb") as f:
        pickle.dump({
            "bitstring_length_bits": len(encoded_bitstring),
            "bitstring": encoded_bitstring,  # note: stored as string
            "shape": (W8, H8),
            "channel_lengths": (len(Y_qcoeffs), len(Cb_qcoeffs), len(Cr_qcoeffs))
        }, f)

    size_entropy = os.path.getsize(entropy_filename)
    print(f"Saved final entropy file: {entropy_filename} ({size_entropy} bytes)")

    #File sizes: original input, pre-entropy, after entropy
    original_size = os.path.getsize(input_path)
    print("\n=== FILE SIZE REPORT ===")
    print(f"Original input file:           {input_path} -> {original_size} bytes ({original_size/1024:.2f} KB)")
    print(f"Pre-entropy quantized coeffs:  pre_entropy_all_qcoeffs_full.pkl -> {size_pre_entropy} bytes ({size_pre_entropy/1024:.2f} KB)")
    print(f"Final entropy-coded file:      {entropy_filename} -> {size_entropy} bytes ({size_entropy/1024:.2f} KB)")

    if size_entropy > 0:
        ratio_pre_to_entropy = size_pre_entropy / size_entropy
        ratio_original_to_entropy = original_size / size_entropy
    else:
        ratio_pre_to_entropy = float('inf')
        ratio_original_to_entropy = float('inf')

    print(f"\nCompression ratio (pre-entropy / after entropy) = {ratio_pre_to_entropy:.2f}x")
    print(f"Compression ratio (original / after entropy)     = {ratio_original_to_entropy:.2f}x")

    # Compute metrics (Y-channel MSE, PSNR, SSIM)
    # Convert original and reconstructed to Y channel arrays for fair comparison
    orig_y = np.array(img_orig.convert("YCbCr").split()[0], dtype=np.uint8)
    recon_y = np.array(img_recon_rgb.convert("YCbCr").split()[0], dtype=np.uint8)

    mse_val = mse(orig_y, recon_y)
    psnr_val = psnr(orig_y, recon_y)
    ssim_val = ssim_index(orig_y, recon_y)

    print("\n=== RECONSTRUCTION METRICS (Y channel) ===")
    print(f"MSE  = {mse_val:.4f}")
    print(f"PSNR = {psnr_val:.4f} dB")
    print(f"SSIM = {ssim_val:.6f}")

except Exception as e:
    print("Error:", e)
