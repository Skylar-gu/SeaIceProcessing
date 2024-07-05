import monte_carlo_bw as mtc
import cv2 as cv
import numpy as np
import rasterio
from shapely import geometry
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import os
import logging
import pickle

def read_image(filepath):
    with rasterio.open(filepath) as f:
        img = f.read(1)
    img = img / np.max(img) * 255
    return img.astype(np.uint8)

def callback(x):
    pass

def canny_edge_opencv(img, low, high):
    _, bin = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    erosion = cv.erode(bin, kernel, iterations=1)
    edges_array = cv.Canny(img, low, high)
    return cv.bitwise_and(edges_array, edges_array, mask=erosion)

def find_contours(edges_arr, area_limit, kernel):
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel)
    dilate = cv.dilate(edges_arr, kernel_ellipse, iterations=1)
    cnts, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    polygon_area_list = []
    for cont in cnts:
        if cv.contourArea(cont) > area_limit:
            polygon = geometry.Polygon(np.squeeze(cont))
            polygons.append(polygon)
            polygon_area_list.append(polygon.area)
    return polygons, polygon_area_list 

def check_inside_contours(polygon_list, polygon_area_list):
    polygons = np.array(polygon_list)
    polygons_area = np.array(polygon_area_list)
    
    valid_polygons = np.array([make_valid(poly) if not poly.is_valid else poly for poly in polygons])

    to_remove = set()
    for i, poly in enumerate(valid_polygons):
        for j, other in enumerate(valid_polygons):
            if i != j and poly.contains(other):
                to_remove.add(j)
    
    mask = np.ones(len(polygons), dtype=bool)
    mask[list(to_remove)] = False
    
    polygons_indep = polygons[mask]
    polygons_area_indep = polygons_area[mask]
    
    return polygons_indep, polygons_area_indep

def log_normal(x, mu, sigma):
    return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))) # Probability Density Function

def find_parameters(filepath, crop=False, crop_size=500):
    global img
    height, width = img.shape
    
    if crop:
        start_h = max(height // 2 - crop_size // 2, 0)
        end_h = min(height // 2 + crop_size // 2, height)
        start_w = max(width // 2 - crop_size // 2, 0)
        end_w = min(width // 2 + crop_size // 2, width)

        img = img[start_h:end_h,start_w:end_w]
    
    cv.namedWindow('image')
    cv.createTrackbar('Low', 'image', 0, 255, callback)
    cv.createTrackbar('High', 'image', 0, 255, callback)
    cv.createTrackbar('Kernel', 'image', 3, 20, callback)
    cv.createTrackbar('Area Limit', 'image', 200, 1000, callback)
    
    while True:
        low = cv.getTrackbarPos('Low', 'image')
        high = cv.getTrackbarPos('High', 'image')
        kernel = cv.getTrackbarPos('Kernel', 'image')
        kernel = kernel if kernel % 2 else kernel + 1
        kernel = (kernel, kernel)
        area_limit = cv.getTrackbarPos('Area Limit', 'image')
        
        edges_arr = canny_edge_opencv(img, low, high)
        polygons, polygon_areas = find_contours(edges_arr, area_limit, kernel)
        #merged_polygons = merge_polygons(polygons)
        polygons_checked, _ = check_inside_contours(polygons, polygon_areas)
        
        results = {}
        results['Low'] = low
        results['High'] = high
        results['Kernel'] = kernel
        results['Area_limit'] = area_limit
    
        display_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for polygon in polygons_checked:
            pts = np.array(polygon.exterior.coords, np.int32).reshape((-1,1,2))
            cv.polylines(display_img, [pts], True, (0,0,255), 1)
        
        # Create edges array image
        edges_display = cv.cvtColor(edges_arr, cv.COLOR_GRAY2BGR)
        
        combined_img = np.concatenate((display_img, edges_display), axis=1)
        
        cv.putText(combined_img, f'Low: {low}, High: {high}, Kernel: {kernel}), Area Limit: {area_limit}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv.putText(combined_img, f'Number of Polygons: {len(polygons_checked)}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv.imshow('image', combined_img)
        
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    cv.destroyAllWindows()
    cv.waitKey(1)
    time.sleep(0.1)
    
    return results

def use_parameters(results):
    global filepath
    low = results['Low']
    high = results['High']
    kernel = results['Kernel']
    area_limit = results['Area_limit']

    img = read_image(filepath)
    edges_arr = canny_edge_opencv(img, low, high)  
    polygons, polygon_areas = find_contours(edges_arr, area_limit, kernel)
    polygons_checked, polygons_area_checked = check_inside_contours(polygons, polygon_areas)
                
    return polygons_checked, polygons_area_checked

def plot_polygons(results, polygons_checked):
    img = read_image(filepath)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    
    for polygon in polygons_checked:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=0.5) # optional: color='red'
    
    low = results['Low']
    high = results['High']
    kernel = results['Kernel']
    area_limit = results['Area_limit']

    title = f'''Image with Polygons
        Low: {low}, High: {high}, Kernel: {kernel}, Area Limit: {area_limit}
        Number of Polygons: {len(polygons_checked)}'''
              
    plt.title(title)
    
    plt.axis('off') 
    plt.tight_layout()
    plt.show()

def save_polygon_img(output_dir, results, polygons_checked):
    global filepath, filename
    img = read_image(filepath)
    plt.figure(figsize=(10, 10))  # Set a fixed figure size for consistency
    plt.imshow(img, cmap='gray')
    
    for polygon in polygons_checked:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            plt.plot(x, y, color='red', linewidth=0.5)

    low = results['Low']
    high = results['High']
    kernel = results['Kernel']
    area_limit = results['Area_limit']
    
    title = f'''{filename[:-4]}
        Low: {low}, High: {high}, Kernel: {kernel}, Area Limit: {area_limit}
        Number of Polygons: {len(polygons_checked)}'''
              
    plt.title(title)
    
    name = f"polygon_plot_{filename}_low_{low}_high_{high}_kernel_{kernel}_area_limit_{area_limit}.pdf"

    filepath = os.path.join(output_dir, name)
    
    plt.savefig(filepath, dpi=1000, bbox_inches='tight')
    plt.close()

def linear(x, m, b):
    return m * x + b

def floe_size_distribution(output_dir, polygons_area_checked, filename, threshold = 100000):
    polygon_areas = np.array(polygons_area_checked)
    print(f'Original Number of Polygons: {len(polygon_areas)}')

    outliers = polygon_areas[polygon_areas > threshold]
    filtered_list = polygon_areas[polygon_areas <= threshold]
    filtered_list = np.sqrt(filtered_list / np.pi)
    print(f'Number of Polygons Plotted: {len(filtered_list)}')

    n, bins, _ = plt.hist(filtered_list, bins=100, edgecolor='black')  # use density = True to normalize the data, n = number of values in each bin
    plt.yscale('log')

    name = f"{filename}_floe_size_distribution.pdf"
    filepath = os.path.join(output_dir, name)
    
    if filename.endswith('.tif'):
        file = filename[:-4]
    else:
        file = filename
    
    title = f'''Floe Size Distribution
    Number of floes plotted: {len(filtered_list)}
    {file}'''

    plt.title(title)
    plt.xlabel('Floe Size (m)')
    plt.ylabel('Density (log scale)')

    plt.grid(True, which="both", ls='-', alpha=0.2)

    # Calculate bin centers
    bin_centers = bins[:-1] + np.diff(bins) / 2

    # Make sure there's no division by 0
    mask = n > 0
    bin_centers = bin_centers[mask]
    n = n[mask]
    
    # Fit a linear distribution to the histogram data
    popt, _ = curve_fit(linear, bin_centers, np.log(n))

    # Plot the fitted curve
    x_interval_for_fit = np.linspace(bins[0], bins[-1], 10000)
    plt.plot(x_interval_for_fit, np.exp(linear(x_interval_for_fit, *popt)), label='Fitted Linear Curve')

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=1000, bbox_inches='tight')
    plt.close()
    
    return outliers

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # PROCESSING FLOES
    logger.info("Starting to process floe images")
    output_dict_floes = {}
    input_directory_floes = '/Users/skylargu/Desktop/selected_images/Floes'
    output_dir_polygons_floes = '/Users/skylargu/Desktop/sea_ice_images/Polygons/Floes'
    output_dir_hist_floes = '/Users/skylargu/Desktop/sea_ice_images/Histograms/Floes'
    
    floe_files = sorted([f for f in os.listdir(input_directory_floes) if f.endswith('.tif')])
    total_floe_files = len(floe_files)

    for idx, filename in enumerate(floe_files, 1):
        logger.info(f"Processing floe image {idx}/{total_floe_files}: {filename}")
        
        filepath = os.path.join(input_directory_floes, filename)
        img = read_image(filepath)
        
        results = {'Low': 50, 'High': 150, 'Kernel': (3,3), 'Area_limit': 700}
        
        polygons_checked, polygons_area_checked = use_parameters(results)
#         save_polygon_img(output_dir_polygons_floes, results, polygons_checked)
#         
        if filename not in output_dict_floes:
            output_dict_floes[filename] = {}
        output_dict_floes[filename]['Polygon_list'] = polygons_checked
        output_dict_floes[filename]['Polygon_areas_list'] = polygons_area_checked
        
        floe_size_distribution(output_dir_hist_floes, polygons_area_checked, filename)
        
        logger.info(f"Completed processing floe image {idx}/{total_floe_files}: {filename}")
    
    logger.info("Finished processing all floe images")

    # PROCESSING LEADS
    logger.info("Starting to process lead images")
    output_dict_leads = {}
    input_directory_leads = '/Users/skylargu/Desktop/selected_images/Leads'
    output_dir_polygons_leads = '/Users/skylargu/Desktop/sea_ice_images/Polygons/Leads'
    
    lead_files = sorted([f for f in os.listdir(input_directory_leads) if f.endswith('.tif')])
    total_lead_files = len(lead_files)

    for idx, filename in enumerate(lead_files, 1):
        logger.info(f"Processing lead image {idx}/{total_lead_files}: {filename}")
        
        filepath = os.path.join(input_directory_leads, filename)
        img = read_image(filepath)
        
        results = {'Low': 50, 'High': 150, 'Kernel': (3,3), 'Area_limit': 1200}
        
        polygons_checked, polygons_area_checked = use_parameters(results)
#       save_polygon_img(output_dir_polygons_leads, results, polygons_checked)
#     
        sat_img, total_px = mtc.process_image(img) 
        bl_px, wh_px = mtc.count_pixels(sat_img)
        sea_ice_conc = mtc.calculate_SIC(bl_px, total_px)
        
        if filename not in output_dict_leads:
            output_dict_leads[filename] = {}
        output_dict_leads[filename]['Polygon_list'] = polygons_checked
        output_dict_leads[filename]['Polygon_areas_list'] = polygons_area_checked
        output_dict_leads[filename]['SIC'] = sea_ice_conc

        logger.info(f"Completed processing lead image {idx}/{total_lead_files}: {filename}")
    
    logger.info("Finished processing all lead images")
    
    # Save the dictionaries
    output_directory = '/Users/skylargu/Desktop/sea_ice_images/Output_Dictionaries'
    os.makedirs(output_directory, exist_ok=True)

    floes_dict_path = os.path.join(output_directory, 'output_dict_floes.pickle')
    leads_dict_path = os.path.join(output_directory, 'output_dict_leads.pickle')

    logger.info("Saving output dictionaries")

    with open(floes_dict_path, 'wb') as f:
        pickle.dump(output_dict_floes, f)
    logger.info(f"Saved floes dictionary to {floes_dict_path}")

    with open(leads_dict_path, 'wb') as f:
        pickle.dump(output_dict_leads, f)
    logger.info(f"Saved leads dictionary to {leads_dict_path}")

    logger.info("Script execution completed")

# To find parameters for a single image and plot it
'''
    filepath = '/Users/skylargu/Desktop/selected_images/Floes/BeaufortSea_2022_03_26.tif'
    img = read_image(filepath)
    results = find_parameters(filepath, crop=True, crop_size=700)
    polygons_checked, polygons_area_checked = use_parameters(results)
    plot_polygons(results, polygons_checked)
'''
