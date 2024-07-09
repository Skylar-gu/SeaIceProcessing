from geopy import distance
from geotiff import GeoTiff

def check_pxl_distance(filepath):
    geotiff = GeoTiff(filepath)

    coord1, coord2= geotiff.get_coords(0,-1), geotiff.get_coords(0,0)
    # get coords in format (lat, lon)
    coord1, coord2 = (coord1[1],coord1[0]), (coord2[1], coord2[0])

    meter_res_pxl = round(geodesic(coord1,coord2).meters,1)
    return meter_res_pxl

