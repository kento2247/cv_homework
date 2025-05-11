import cv2
import os

import numpy as np

def apply_bilateral_filter(image: np.ndarray) -> np.ndarray:
    # Apply bilateral filter once
    filtered_image = cv2.bilateralFilter(image, d=0, sigmaColor=10, sigmaSpace=30)
    
    return filtered_image

def apply_guided_filter(image: np.ndarray, radius: int = 10, eps: float = 1e-2) -> np.ndarray:

    # Convert image to float32
    image_float = image.astype(np.float32) / 255.0

    # Apply guided filter
    guided_image = cv2.ximgproc.guidedFilter(guide=image_float, src=image_float, radius=radius, eps=eps)
    
    return (guided_image * 255).astype(np.uint8)

if __name__ == "__main__":
    # Example usage
    N=5
    bilateral_image = cv2.imread("ImageProcessing/images/target.jpg")
    guided_image = cv2.imread("ImageProcessing/images/target.jpg")
    import time
    
    # Bilateral filter processing
    print("Processing bilateral filter...")
    bilateral_start_time = time.time()
    for i in range(N):
        start_time = time.time()
        bilateral_image = apply_bilateral_filter(bilateral_image)
        bilateral_save_path = os.path.join("ImageProcessing/images", f"bilateral_{i}.jpg")
        cv2.imwrite(bilateral_save_path, bilateral_image)
        end_time = time.time()
        print(f"Saved filtered image to {bilateral_save_path}")
        print(f"Bilateral filter iteration {i+1} took {end_time - start_time:.4f} seconds")
    bilateral_total_time = time.time() - bilateral_start_time
    print(f"Total bilateral filter processing time: {bilateral_total_time:.4f} seconds")
    
    # Guided filter processing
    print("\nProcessing guided filter...")
    guided_start_time = time.time()
    for i in range(N):
        start_time = time.time()
        guided_image = apply_guided_filter(guided_image)
        guided_save_path = os.path.join("ImageProcessing/images", f"guided_{i}.jpg")
        cv2.imwrite(guided_save_path, guided_image)
        end_time = time.time()
        print(f"Saved guided filtered image to {guided_save_path}")
        print(f"Guided filter iteration {i+1} took {end_time - start_time:.4f} seconds")
    guided_total_time = time.time() - guided_start_time
    print(f"Total guided filter processing time: {guided_total_time:.4f} seconds")
