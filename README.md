# SeaIceProcessing
 
Analyses satellite images of sea ice to automatically detect and measure ice floes. Uses computer vision to find the edges of ice pieces, convert them to polygons, and work out statistics like size distributions.

### Main Scripts
get_floes_polygon: The core detection algorithm. Takes a satellite image and returns polygons for each ice floe it finds.
plot_polygons_and_histograms.py: Batch processes multiple images. Has an interactive mode where you can tune the detection parameters with sliders and see results in real-time.
calculate_sic.py: Calculates sea ice concentration 
plot_ice_on_map.py: Overlays the ice images onto proper maps with pressure data from ERA5.

### Quick Example
```import get_floes_polygon as gfp

img = gfp.read_image('path/to/satellite_image.tif')
polygons, areas = gfp.get_polygons(img, low=50, high=150, 
                                    kernel=(3,3), area_limit=300)```
