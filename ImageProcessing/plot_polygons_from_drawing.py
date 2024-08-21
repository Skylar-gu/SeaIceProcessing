import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from shapely import geometry
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiPoint
from scipy.optimize import curve_fit
from geopy.distance import geodesic
from geotiff import GeoTiff
import rasterio
import os

def read_image(filepath):
    with rasterio.open(filepath) as f:
        img = f.read(1)
    img = img / np.max(img) * 255
    return img.astype(np.uint8)
  
# Function to read and convert PDF to grayscale image
def read_pdf(filepath):
    pdf = convert_from_path(filepath)
    img_np = np.array(pdf)
    img_np = np.squeeze(img_np, axis=0)
    
    if img_np.ndim == 3:
        img_gray = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
    else:
        img_gray = img_np

    img_gray = img_gray.astype(np.uint8)

    return img_gray

# Function to find contours and convert them to polygons
def find_contours(edges_arr):
    cnts, _ = cv.findContours(edges_arr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    polygon_area_list = []
    for cont in cnts:
        if cv.contourArea(cont) > 0: 
            polygon = Polygon(np.squeeze(cont) * 10.077669902912621)
            polygons.append(polygon)
            polygon_area_list.append(polygon.area)
    return polygons, polygon_area_list 

# Function to plot the polygons
def plot_polygons(polygons, image):
    plt.figure(figsize=(20, 20))
    plt.imshow(image, cmap='gray')
    
    for polygon in polygons:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=0.3)
    
    plt.axis('off') 
    plt.tight_layout()
    plt.show()

# Filepath to the PDF
filepath = '/Users/skylargu/Desktop/BeaufortSea_2021_04_28_drawing.pdf'

# Read PDF and convert to grayscale image
img = read_pdf(filepath)
img = img[270:1300, 120:1596]

# Apply Canny edge detection
edges = cv.Canny(img, 50, 150)

# Find contours and convert them to polygons
polygons, polygon_areas = find_contours(edges)

tiff_img = read_image("/Users/skylargu/Desktop/selected_images/Floes/BeaufortSea_2021_04_28.tif")
# Plot the polygons
plot_polygons(polygons, tiff_img)
