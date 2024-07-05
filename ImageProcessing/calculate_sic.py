import numpy as np
import cv2
import random
import rasterio
from shapely import geometry
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time

def read_image(filepath):
    with rasterio.open(filepath) as f:
        img = f.read(1)
    img = img / np.max(img) * 255
    return img.astype(np.uint8)

def calculate_black_percentage(black_pixel, total_pixels):
    return round((black_pixel / total_pixels) * 100, 2)

def calculate_SIC(black_pixel, total_pixels):
    return round(((total_pixels - black_pixel)/ total_pixels) * 100, 2)

def count_pixels(img):
    black_pixel = np.sum(img <= 100)
    white_pixel = np.sum(img > 100)
    return black_pixel, white_pixel

def process_image(img):
    polygon = np.where(img==0, np.nan, img) # change elements in black border to nan
    nan_mask = np.isnan(polygon) # returns a boolean array
    sat_img = img[~nan_mask] # ~nan.mask = inverted nan_mask, boolean indexing flattens the 2D array into a 1D one
    return sat_img, sat_img.shape[0]

   
if __name__ == "__main__":
    filepath = '/Users/skylargu/Desktop/selected_images/BeaufortSea_2021_04_28.tif'
    
    img = read_image(filepath)
    # Image as a numpy array
    
    sat_img, total_px = process_image(img) # number of pixels part of the satellite image
    print(sat_img)
    
    iterations = [100,1000,10000,1000000]
    
    # Count of black and white pixel in the random sampling
    count_white = [0]*4 # = [0,0,0,0]
    count_black = [0]*4
    
    bl_px,wh_px = count_pixels(sat_img)
    
    print("Total number of black pixels(brute force):",bl_px)
    print("Total number of white pixels(brute force):",wh_px)
    print("Percentage of black pixels(brute force):",calculate_black_percentage(bl_px,total_px))
    
    # Getting "iteration" number of pixels
    for i, iteration in enumerate(iterations):
        sample = np.random.choice(sat_img, size=iteration, replace=False)
        count_black[i] = np.sum(sample <= 100)
        count_white[i] = iteration - count_black[i]
    
        print(f"\n For {iteration} iterations:") 
        print(f"Estimated number of black pixels:{count_black[i]}")
        print(f"Estimated number of white pixels:{count_white[i]}")
        print(f"Estimated percentage of black pixels:", calculate_black_percentage(count_black[i], count_black[i]+ count_white[i]))
