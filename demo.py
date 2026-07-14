import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def make_sample_image(path, size=(300, 220)):
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        arr[y, :, 0] = int(60 + (255 - 60) * t)
        arr[y, :, 1] = int(30 + (180 - 30) * t)
        arr[y, :, 2] = int(80 + (120 - 80) * (1 - t))
    Image.fromarray(arr).save(path)
    print(f"No image found -- generated a sample gradient at '{path}'")



def load_image_as_vectors(path, size=(300, 220)):
    if not os.path.exists(path):
        make_sample_image(path, size)
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype(np.float64)   
    h, w, _ = arr.shape
    vectors = arr.reshape(-1, 3)                
    return arr, vectors, (h, w)



def quantize_value(v, n_levels, lo, hi):
    """Snap v onto n_levels evenly spaced points between lo and hi."""
    if n_levels <= 1:
        return np.full_like(v, (lo + hi) / 2)
    step = (hi - lo) / (n_levels - 1)
    idx = np.round((v - lo) / step)
    idx = np.clip(idx, 0, n_levels - 1)
    return lo + idx * step


def direct_quantize(vectors, n_levels):
    return quantize_value(vectors, n_levels, 0, 255)



def find_rotation(vectors):
    mean = vectors.mean(axis=0)                       
    centered = vectors - mean                          
    cov = np.cov(centered, rowvar=False)                

    eigvals, eigvecs = np.linalg.eigh(cov)              
    order = np.argsort(eigvals)[::-1]                   
    R = eigvecs[:, order]                                
    return R, mean




def rotate_quantize_rotate_back(vectors, n_levels, R, mean):
    centered = vectors - mean
    rotated = centered @ R                              

    lo = rotated.min(axis=0)
    hi = rotated.max(axis=0)
    rotated_q = quantize_value(rotated, n_levels, lo, hi)

    back = rotated_q @ R.T + mean                       
    return back



def mse(a, b):
    return np.mean((a - b) ** 2)


def psnr(m):
    if m <= 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(m))



def run_demo(image_path, n_levels=3):
    arr, vectors, (h, w) = load_image_as_vectors(image_path)

    
    direct = direct_quantize(vectors, n_levels)

    
    R, mean = find_rotation(vectors)
    rotated_back = rotate_quantize_rotate_back(vectors, n_levels, R, mean)

    mse_direct = mse(vectors, direct)
    mse_rotate = mse(vectors, rotated_back)

    print(f"Color budget: {n_levels} levels/channel  (~{n_levels**3} colors)")
    print(f"Direct quantization    -> MSE: {mse_direct:.2f} | PSNR: {psnr(mse_direct):.1f} dB")
    print(f"Rotate->quantize->back -> MSE: {mse_rotate:.2f} | PSNR: {psnr(mse_rotate):.1f} dB")
    winner = "ROTATION" if mse_rotate < mse_direct else "DIRECT"
    print(f"Winner (lower error): {winner}")

    
    def to_img(v):
        return np.clip(v, 0, 255).astype(np.uint8).reshape(h, w, 3)

    orig_img = to_img(vectors)
    direct_img = to_img(direct)
    rotate_img = to_img(rotated_back)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [orig_img, direct_img, rotate_img],
        ["Original", f"Direct quantize\nMSE={mse_direct:.1f}", f"Rotate->quantize->back\nMSE={mse_rotate:.1f}"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("rotate_quantize_result.png", dpi=150)
    print("\nSaved comparison figure to rotate_quantize_result.png")
    plt.show()


if __name__ == "__main__":
    
    run_demo("image.png", n_levels=3)
