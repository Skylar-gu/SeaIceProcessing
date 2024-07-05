import plot_polygons_and_historgrams as pph
import pickle
import os

total_area_list = []

with open("/Users/skylargu/Desktop/sea_ice_images/Output_Dictionaries/output_dict_floes.pickle", "rb") as f: #wb/rb: write/read binary
    output_dict_floes = pickle.load(f)

input_directory_floes = '/Users/skylargu/Desktop/selected_images/Floes'

floe_files = sorted([f for f in os.listdir(input_directory_floes) if f != '.DS_Store'])

for idx, filename in enumerate(floe_files, 1):
    filepath = os.path.join(input_directory_floes, filename)
    img = pph.read_image(filepath)
    
    polygons_area_checked = output_dict_floes[filename]['Polygon_areas_list']

    total_area_list.extend(polygons_area_checked)
    

output_dir = '/Users/skylargu/Desktop/sea_ice_images/Histograms'
filename = 'total'
pph.floe_size_distribution(output_dir, total_area_list, filename)
