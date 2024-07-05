import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_image(filepath):
    with rasterio.open(filepath) as f:
        img = f.read(1)
    img = img / np.max(img) * 255
    return img.astype(np.uint8)

def tiff2pdf(filepath, output_dir, filename):
    img = read_image(filepath)

    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')

    filename = filename+".pdf"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=1000, bbox_inches='tight')
    plt.close()

input_directory = '/storage/sgu/selected_images'
output_dir = '/storage/sgu/selected_images_pdf'

for filename in sorted(os.listdir(input_directory)):
    if filename == '.DS_Store':
        continue
    filepath = os.path.join(input_directory, filename)
    tiff2pdf(filepath, output_dir, filename)
