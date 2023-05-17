"""
import folium 
import h3 
import geopandas as gpd
import pandas

# Ottieni gli esagoni di H3 di livello 8 
h3_level = 8 
hexagons = h3.h3_to_geo_boundary(h3.geo_to_h3(37.7749, -122.4194, h3_level)) 

gdf = gpd.GeoDataFrame(geometry=hexagons, crs='EPSG:4326') 

base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with

print(gdf)"""

import geopandas as gpd
from shapely.geometry import Polygon
import h3 
import matplotlib.pyplot as plt
# Get H3 hexagons at level 8 h3_level = 8 
hexagons = h3.h3_to_geo_boundary(h3.geo_to_h3(37.7749, -122.4194, 8)) 
# # Create a GeoDataFrame 
gdf = gpd.GeoDataFrame(geometry=[Polygon(hexagons)], crs='EPSG:4326') 
# # Save the GeoDataFrame to a file 
gdf.to_file('prova.shp')


gdf.plot(aspect='equal') # Display the plot import matplotlib.pyplot as plt 
plt.show()













