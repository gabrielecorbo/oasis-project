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

# Get H3 hexagons at level 8
h3_level = 8
center_lat, center_lon = 37.7749, -122.4194
hexagons = list(h3.k_ring(h3.geo_to_h3(center_lat, center_lon, h3_level), 10))  # Convert set to list

# Create a GeoDataFrame with a single polygon
gdf = gpd.GeoDataFrame(geometry=[Polygon(h3.h3_to_geo_boundary(hexagons[0], True))], crs='EPSG:4326')

# Append additional polygons to the GeoDataFrame
for hexagon in hexagons[1:]:
    gdf = gdf.append({'geometry': Polygon(h3.h3_to_geo_boundary(hexagon, True))}, ignore_index=True)

# Plot the GeoDataFrame
gdf.plot()

# Display the plot
import matplotlib.pyplot as plt
plt.show()








