import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import rasterio
import os
import csv
import ast
from netCDF4 import Dataset as netcdfile
import tifffile
from PIL import Image

# Change filepaths accordingly
input_directory = '/Users/skylargu/Desktop/selected_images'
output_directory = '/Users/skylargu/Desktop/MSLP_maps'

def find_coordinates_and_centroid(filename):
    csv_file = '/Users/skylargu/Desktop/Updated_Coordinates.csv'

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['filename'] == filename:
                coordinates = ast.literal_eval(row['coordinates'])
                centroid = ast.literal_eval(row['centroid'])
                return coordinates[0], centroid

    raise ValueError("Filename not found")

def transform_string(input_str):
    parts = input_str.split("_")
    transformed = f"{parts[0]} {parts[1]}/{parts[2]}/{parts[3]} MSLP"
    return transformed

filename = 'MClureStrait_2024_05_05.tif'

filepath = os.path.join(input_directory, filename)


image = rasterio.open(filepath)
size = image.count
image = image.read()

if size > 2:
    imarray = np.array(image[0])
else:
    imarray = np.array(image)

imarray = (imarray - imarray.min()) / (imarray.max() - imarray.min())

# Find coords and centroid
coords, centroid = find_coordinates_and_centroid(filename)
lons, lats = zip(*coords)

# find centroid
center_lon = centroid[0]
center_lat = centroid[1]

# Find the right MSLP file
length = len(filename)
date = filename[length-14:length-4]
year, month, day = date.split("_")

SLP_filepath = f"/Users/skylargu/Desktop/ERA5_MSLP_2019-2024/ERA5_{year}.nc"
dataset = netcdfile(SLP_filepath)

days = 0
if month == "03":
    days = int(day) - 1
elif month == "04":
    days = 31 + int(day) - 1
elif month == "05":
    days = 61 + int(day) - 1

    # load all variables
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]
time = dataset.variables['time'][:]
mslp = dataset.variables['msl'][:]

# Select the first time step for plotting
msl_data = mslp[days, :, :]
msl_data= msl_data/100 - 1000
'''
    # Convert the data to float (if needed, based on the scaling factors)
if hasattr(dataset.variables['msl'], 'scale_factor'):
    scale_factor = dataset.variables['msl'].scale_factor
    add_offset = dataset.variables['msl'].add_offset
    msl_data = msl_data * scale_factor + add_offset
'''
# from previous cell
proj = ccrs.TransverseMercator(central_longitude=center_lon, central_latitude=center_lat)
data_crs = ccrs.PlateCarree()

# Set up the figure
fig = plt.figure(figsize=(20, 10))

left = 0.1
bottom = 0.1
width = 0.4
height = 0.8
rect = [left, bottom, width, height]
axi = plt.axes(rect, projection=proj)

axi.coastlines()
axi.add_feature(cartopy.feature.LAND)
extent = [min(lons) -10, max(lons) + 10, min(lats) - 3, max(lats) + 3]
extent_im = [min(lons), max(lons), min(lats), max(lats)]
axi.set_extent(extent, crs=data_crs)

levels = np.arange(np.floor(msl_data.min()), np.ceil(msl_data.max()), 2)
#8 mili bars = 800 pa
#/1000, -1000, around 1000, high 1020
# Plot the MSL data
msl_plot = axi.contour(longitude, latitude, msl_data, levels=levels, transform=ccrs.PlateCarree(), cmap='viridis')

'''axi.clabel(msl_plot, inline=True, fontsize=10, fmt='%1.0f hPa', inline_spacing=3)
cbar = plt.colorbar(msl_plot, orientation='horizontal', pad=0.05)
cbar.set_label('Mean Sea Level Pressure (mbar)')'''

# Plot the image
axi.imshow(imarray, origin='upper', extent=extent_im, transform=ccrs.PlateCarree(), cmap='gray', zorder=2)

sub_ax_extent =[min(lons) -0.2, max(lons) + 0.2, min(lats) - 0.1, max(lats) + 0.05]
left = 0.45
bottom = 0.15
width = 0.45
height = 0.7
rect = [left, bottom, width, height]

sub_ax = plt.axes(rect, projection=proj) #left, bottom, width, height
sub_ax.set_extent(sub_ax_extent, crs=data_crs)
#sub_ax.set_aspect('equal', adjustable='box')

x0 ,x1, y0, y1 = sub_ax.get_extent()
rectangle = plt.Rectangle((x0 - 0.05, y0 - 0.05), x1 - x0 + 0.1, y1-y0 + 0.1,
                          linewidth=1.5, edgecolor='rebeccapurple', facecolor='none', transform=data_crs)
axi.add_patch(rectangle)

sub_ax.coastlines()
sub_ax.imshow(imarray, origin='upper', extent=extent_im, transform=ccrs.PlateCarree(), cmap='gray')

title = transform_string(filename[:-4])
axi.set_title(title)

# Show the plot
plt.show()
'''output_filepath = os.path.join(output_directory, filename[:-4])
    plt.savefig(f"{output_filepath}_map.png")'''

# Close the plot to move on to the next image
plt.close()
