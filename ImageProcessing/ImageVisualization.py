import tifffile
from matplotlib import pyplot as plt
import numpy as np
import os

# Directories containing the images
input_directory = '/Users/skylargu/Desktop/selected_images'
output_directory = '/Users/skylargu/Desktop/Selected_part2'

#Note: Everything below is inside the for loop
# Iterate through each file in the directory
for filename in sorted(os.listdir(input_directory)):
    filepath = os.path.join(input_directory, filename)
    # filepath1 = '/Users/skylargu/Desktop/sea_ice_images/Utqiagvik_2024_03_15.tif'
    try: 
        image = tifffile.imread(filepath)
    except OSError:
        print(f"{filename} cannot be read")
        continue
    except:
        print(f"{filename} cannot be read (possibly invalid tiff file)")
        continue

    # Print the size of the image
    size = image.shape
    print(f"Size of {filename}: {size}")
        
    # If the image has more than 2 dimensions, take the first channel
    if len(size) > 2:
        imarray = np.array(image[0])
        print(f"Shape of the first channel: {imarray.shape}")
    else:
        imarray = np.array(image)
        print(f"Shape of the image: {imarray.shape}")
        
    # To select a specific region (of the image) if necessary
    # subset = imarray[0:500, 0:100]

    # Print the maximum and minimum values in the array
    max_value = imarray.max()
    min_value = imarray.min()
    print(f"Maximum value: {max_value}")
    print(f"Minimum value: {min_value}")

    # Plot the array
    plt.figure(figsize = (10,10))
    plt.imshow(imarray, cmap='gray')
    plt.title(f"Image: {filename}")
    plt.colorbar()
    plt.show()

    save_image = input("Save image? (y/n): ").strip().lower()
    
    if save_image.lower() == 'y':
        # Save the figure to the output directory as a TIFF file
        output_filepath = os.path.join(output_directory, f'dupl_ {filename}')
        plt.savefig(output_filepath, format='tiff')
        print(f"Image saved as: {output_filepath}")
    
    if save_image.lower() == 'repeat':
        plt.figure(figsize = (10,10))
        plt.imshow(imarray, cmap='gray')
        plt.title(f"Image: {filename}")
        plt.colorbar()
        plt.show()
        
     # Close the plot to move on to the next image
    plt.close()
