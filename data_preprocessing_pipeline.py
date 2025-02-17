
import torch
from torchvision import transforms
import numpy as np
import cv2
import numpy as np
from PIL import Image
from skimage import exposure
import glob
import os

unique_colors = [
[0, 0, 0],
[128, 0, 128],
[128, 0, 0],
[0, 128, 0],
[128, 128, 0],
[0, 0, 128],
[0, 128, 128],
[64, 0, 0],
[128, 128, 128],
[192, 0, 0]]
unique_colors = np.array(unique_colors)


def img_normalization(image, normalization_pipeline=[]):
    if isinstance(image, Image.Image):
        image = np.array(image)
    img_array = np.array(image, dtype=np.uint8)

    normlaization_method_dict={
        'min_max': lambda img: (img - img.min()) * 255.0 / (img.max() - img.min()),
        'histogram_matching': lambda img, reference_image: exposure.match_histograms(img, reference_image, multichannel=True),
        'medianblur': lambda img: cv2.medianBlur(img, 5),
        '255clip': lambda img: np.uint8(np.clip(img, 0, 255)),
        'zscore': lambda img: (img - img.mean()) / img.std(),
        'percentileoutlier_remove': lambda img: np.clip(img, np.percentile(img, 2), np.percentile(img, 98)),
    }

    #Start with Array Image (Assume Image is RGB 3xwidthxheight)
    for normalization_step in normalization_pipeline:
        img_array = normlaization_method_dict[normalization_step](img_array)

    result_image = Image.fromarray(img_array)
    return result_image

def get_paths(input_path):
    input_path  =  glob.glob(input_path)
    input_path.sort()
    print(input_path)
    return input_path


# output_dir = 'data/output'
# os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
# output_filename = 'output_image.png'
# output_path = os.path.join(output_dir, output_filename)

# # Save the image as a PNG file.
# example_image.save(output_path, format='PNG')

# print(f"Image saved to {output_path}")

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    counter = 1

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.png'):
            input_path = os.path.join(input_dir, file_name)
            try:
                # Load the image (you can change to .convert('L') if you want grayscale)
                img = Image.open(input_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {input_path}: {e}")
                continue

            # Apply image normalization
            norm_img = img_normalization(img)
            
            # Define the output path and save the images as PNG
            output_path = os.path.join(output_dir, file_name)
            try:
                norm_img.save(output_path, format='PNG')
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error saving {output_path}: {e}")

            counter +=1
            if counter > 5:
                break

if __name__ == '__main__':
    # Set the input and output directories (adjust the paths as needed)
    input_directory = r"C:\path\to\input\directory"
    output_directory = r"C:\path\to\output\directory"
    
    process_images(input_directory, output_directory)

