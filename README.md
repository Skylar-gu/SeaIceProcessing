# SeaIceProcessing
 
Analyses satellite images of sea ice to automatically detect and measure ice floes. Uses computer vision to find the edges of ice pieces, convert them to polygons, and work out statistics like size distributions.

### Main Scripts
- **`get_floes_polygon.py`**  
  Core detection algorithm.  
  Takes a satellite image as input and returns polygons corresponding to individual ice floes.

- **`plot_polygons_and_histograms.py`**  
  Batch processes multiple images.  
  Includes an interactive mode with sliders for tuning detection parameters and visualizing results in real time.

- **`calculate_sic.py`**  
  Computes sea ice concentration (SIC) from detected floes.

- **`plot_ice_on_map.py`**  
  Overlays detected ice floes onto georeferenced maps, including pressure data from ERA5.

### Methodological Process:
1. Convert image to binary (ice vs not-ice)
2. Find edges using Canny edge detection
3. Turn edges into closed polygons
4. Filter out bad polygons (too small, nested inside others, etc.)
5. Check that polygon centres are actually on ice
6. Calculate areas and plot size distributions

### Example Usage
```
import get_floes_polygon as gfp

img = gfp.read_image('path/to/satellite_image.tif')
polygons, areas = gfp.get_polygons(img, low=50, high=150, 
                                    kernel=(3,3), area_limit=300)
```

### Notes:
- Detection parameters need tweaking for different lighting conditions
- Assumes floes are roughly circular when converting area to size
- Minimum detectable size depends on image resolution (usually 10-50m per pixel)
