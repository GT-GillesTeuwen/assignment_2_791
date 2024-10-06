import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Paths to the directories
original_dir = 'images'
results_dirs = ['results/kapur/sa/k2', 'results/kapur/sa/k3', 'results/kapur/sa/k4', 'results/kapur/sa/k5']

# Function to calculate SSIM and PSNR
def calculate_metrics(original_path, result_path):
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    result_img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if both images are successfully loaded
    if original_img is None or result_img is None:
        print(f"Could not open one of the images: {original_path} or {result_path}")
        return None, None

    # Calculate SSIM and PSNR
    ssim_value = ssim(original_img, result_img)
    psnr_value = psnr(original_img, result_img)
    
    return ssim_value, psnr_value

# Store results for presentation
results_table = []

# Loop through each directory and compare images
for results_dir in results_dirs:
    for original_image_name in os.listdir(original_dir):
        if original_image_name.endswith('.jpg'):
            original_image_path = os.path.join(original_dir, original_image_name)
            # Find the corresponding result image
            base_name = os.path.splitext(original_image_name)[0]
            
            for result_image_name in os.listdir(results_dir):
                if result_image_name.startswith(base_name) and result_image_name.endswith('.png'):
                    result_image_path = os.path.join(results_dir, result_image_name)
                    ssim_value, psnr_value = calculate_metrics(original_image_path, result_image_path)
                    
                    if ssim_value is not None and psnr_value is not None:
                        results_table.append([original_image_name, result_image_name, ssim_value, psnr_value])

# Sort the results table by the 'Result Image' column (index 1)
results_table.sort(key=lambda x: x[1])

# Prepare output for easy copying to Google Sheets
ssim_line = "SSIM\n"
psnr_line = "PSNR\n"
for row in results_table:
    original_image_name, result_image_name, ssim_value, psnr_value = row
    ssim_line += f"{ssim_value:.4f}\n\n"
    psnr_line += f"{psnr_value:.4f}\n\n"

# Print the formatted lines
print(ssim_line)
print(psnr_line)
