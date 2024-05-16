import cv2
import numpy as np
import os
from PIL import Image


def apply_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image
def random_downsample(image):
    height, width = image.shape[:2]
    scale_factor = np.random.uniform(0.5, 0.9)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    interpolation_methods = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
    interpolation = np.random.choice(interpolation_methods)
    downsampled_image = cv2.resize(image, new_size, interpolation=interpolation)
    return downsampled_image
def process_image(image_path, output_path, kernel_sizes):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    kernel_size = np.random.choice(kernel_sizes)
    blurred_image = apply_blur(image, kernel_size)
    downsampled_image = random_downsample(blurred_image)
    cv2.imwrite(output_path, downsampled_image)
    print(f"Processed image saved to {output_path}")
def process_images_in_folder(input_folder, output_folder, kernel_sizes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            process_image(input_path, output_path, kernel_sizes)

input_folder = 'path_to_high_res_images'
output_folder = 'path_to_processed_images'
kernel_sizes = [3, 5, 7, 9]
process_images_in_folder(input_folder, output_folder, kernel_sizes)
