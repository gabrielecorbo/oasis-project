import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shapefile as shp
import pyproj
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
import mapclassify as mc
from scipy import ndimage
import geopandas as gpd



desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


#locate manchester traffic data
data_path = os.getcwd()+'\\csv_files\\manchester_traffic_data.csv'
data = pd.read_csv(data_path)
raw_traffic_df = pd.DataFrame(data=data)
print(raw_traffic_df.info())


#importo EV charging station
data_path = os.getcwd()+'\\csv_files\\mean_car_count_per_grid.csv'
data = pd.read_csv(data_path)
mean_car_count = pd.DataFrame(data=data)



# Only use traffic data from the year 2019 and drop useless columns
raw_traffic_df = raw_traffic_df[raw_traffic_df['year'] == 2019]
raw_traffic_df = raw_traffic_df.drop(labels=['region_id', 'region_name', 'local_authority_id', 'local_authority_name',
                                             'start_junction_road_name', 'end_junction_road_name', 'link_length_miles',
                                             'estimation_method', 'estimation_method_detailed', 'pedal_cycles', 'two_wheeled_motor_vehicles',
                                             'buses_and_coaches','lgvs', 'hgvs_2_rigid_axle', 'hgvs_3_rigid_axle',
                                             'hgvs_4_or_more_rigid_axle', 'hgvs_3_or_4_articulated_axle', 'hgvs_5_articulated_axle',
                                             'hgvs_6_articulated_axle' ,'all_hgvs','all_motor_vehicles','year','road_name','road_type',
                                             'link_length_km','count_point_id'], axis=1)


# Add a coordinate column to the dataframe and convert to UK EPSG:27700 (meters)
#proj = pyproj.Transformer.from_crs(4326, 27700, always_xy=True)
#x1, y1 = (raw_traffic_df['longitude'], raw_traffic_df['latitude'])
#x2, y2 = proj.transform(x1, y1)
#x2, y2 = (pd.DataFrame(x2, columns=['horizontal']), pd.DataFrame(y2, columns=['vertical']))
#raw_traffic_df = pd.concat([raw_traffic_df, x2, y2], axis=1)
#
def point_df_to_gdf(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['easting', 'northing']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df



traffic_points_gdf = point_df_to_gdf(raw_traffic_df)
#traffic_points_gdf = raw_traffic_df

print(traffic_points_gdf.head())
#traffic_points_gdf = traffic_points_gdf.set_crs(crs="EPSG:4326")
print('Traffic CRS', '\n', traffic_points_gdf.crs)
traffic_points_gdf.to_file(os.getcwd()+'\\shapefiles\\traffic_points.shp')

#sf_traffic_points

# adding roads to the plot of the traffic measurement points
shp_path_roads_1 = os.getcwd()+'\\shapefiles\\SD_Region.shp'
shp_path_roads_2 = os.getcwd()+'\\shapefiles\\SJ_Region.shp'
sf_roads_1, sf_roads_2 = (shp.Reader(shp_path_roads_1), shp.Reader(shp_path_roads_2, encoding='windows-1252'))

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords'
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    records = [y[:] for y in sf.records()]
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df


df_roads_1, df_roads_2 = (read_shapefile(sf_roads_1), read_shapefile(sf_roads_2))
df_roads = pd.concat([df_roads_1, df_roads_2])  # Combine road dataframes into single dataframe

# Select whether to include motorways or not (currently excludes motorways)
df_roads_exc_mtrwy = df_roads[~df_roads['class'].str.contains('Motorway')]
df_roads_exc_mtrwy['coords'] = df_roads_exc_mtrwy['coords'].apply(LineString)
df_roads_exc_mtrwy = gpd.GeoDataFrame(df_roads_exc_mtrwy, geometry='coords')

y_lim = (393500,401000)                                    # y coordinates (boundaries of city of Manchester)
x_lim = (382500,389500)                                    # x coordinates (boundaries of city of Manchester)
x1_y1 = (-2.2648971967997866,53.437999025519474)             # latitudes (boundaries of city of Manchester)
x2_y2 = (-2.1597774081293526,53.5055991531199)            # longitudes (boundaries of city of Manchester)
#inProj = pyproj.CRS(init='epsg:27700')
#outProj = pyproj.CRS(init='epsg:4326')
#x1, y1 = x_lim[0], y_lim[0]
#x2, y2 = x_lim[1], y_lim[1]
#x1, y1 = pyproj.transform(inProj, outProj, x1, y1)
#x2, y2 = pyproj.transform(inProj, outProj, x2, y2)
#print(x1, y1)
#print(x2, y2)
#rect mio per clippare punti
rect=Polygon([(x_lim[0],y_lim[0]),(x_lim[0],y_lim[1]),(x_lim[1],y_lim[1]),(x_lim[1],y_lim[0]),(x_lim[0],y_lim[0])])
rect_gdf=gpd.GeoDataFrame([1], geometry = [rect], crs=27700)
traffic_points_clip=traffic_points_gdf.clip(rect)
traffic_points_clip.to_file(os.getcwd()+'\\shapefiles\\traffic_points_clip.shp')
df_roads_exc_mtrwy_clip=df_roads_exc_mtrwy.clip(rect)
#df_roads_exc_mtrwy_clip.to_file(os.getcwd()+'\\shapefiles\\map.shp')
#
base = df_roads_exc_mtrwy.plot(figsize=(12, 8), color='deepskyblue', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)
#plt.scatter(x=traffic_points_gdf.easting,y=traffic_points_gdf.northing,c=traffic_points_gdf.cars_and_taxis, cmap='plasma', s=15, zorder=10)

#traffic_points_gdf.plot(ax=base, x='easting', y='northing', c='cars_and_taxis', cmap='viridis', kind='scatter', s=35, zorder=10)
traffic_points_clip.plot(ax=base, x='easting', y='northing', c='cars_and_taxis', cmap='viridis', kind='scatter', s=35, zorder=10)

#traffic_points_gdf.plot(ax=base, x='easting', y='northing', cmap='viridis', kind='scatter', s=7, zorder=10)


def point_grid(y_min, y_max, x_min, x_max):
    """This function takes the coordinate limits and creates a regular grid
    across the area"""

    step_size = 500     # Distance in meters
    gridpoints = []

    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            p = (x, y)
            gridpoints.append(p)
            y += step_size
        x += step_size

    grid_df = pd.DataFrame(data=gridpoints, columns=['x', 'y'])
    plt.scatter(grid_df['x'], grid_df['y'], color='maroon', s=2)
    # open the file in the write mode
    # with open('/optimise_EV_location/gridpoints.csv', 'w') as csv_file:
    #     # create the csv writer
    #     csv_file.write('hor;vert\n')
    #     for p in gridpoints:
    #         csv_file.write('{:f};{:f}\n'.format(p.x, p.y))

def polygon_grid(ymin, ymax, xmin, xmax):
    """This function takes the coordinate limits and creates a polygon grid
    across the area"""
    
    height = 500
    width = 500

    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + height, height))

    polygons = []
    colori=[]
    tot_traffic=[]
    for x in cols[:-1]:
        for y in rows[:-1]:
            rect_i=Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
            polygons.append(rect_i)
            parziale=traffic_points_gdf.clip(rect_i)["cars_and_taxis"].sum()
            tot_traffic.append(parziale)
            if parziale <8830:
                colori.append(1)
            elif (parziale<22200):
                colori.append(2)
            elif (parziale<60000):
                colori.append(3)
            else:
                colori.append(4)
            
            
            
    print("CIAOOOOOO")
    print(tot_traffic)
    print(pd.Series(tot_traffic).describe())
    tot_traffic=np.array(tot_traffic)
    tot_traffic=tot_traffic.reshape(len(rows)-1,len(cols)-1)
    #print(tot_traffic)
    
    colori=np.array(colori)
    colori=colori.reshape(len(rows)-1,len(cols)-1)
    
    poly_grid = gpd.GeoDataFrame({'geometry': polygons})
    poly_grid.plot(ax=base, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
    sns.color_palette("Blues", as_cmap=True)
 
#    base2=sns.heatmap(colori, cmap='Blues')
#    poly_grid.plot(ax=base2, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
#    poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid.shp')
#    
    
#    tot_traffic=np.array(mean_car_count["cars_and_taxis_mean"])
#    tot_traffic=tot_traffic.reshape(17,16)
#    print(tot_traffic)
#    poly_grid = gpd.GeoDataFrame({'geometry': polygons})
#    poly_grid.plot(ax=base, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
#    sns.color_palette("Blues", as_cmap=True)
# 
#    base2=sns.heatmap(tot_traffic, cmap='Blues')
#    poly_grid.plot(ax=base2, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
#    poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid.shp')
    
    

#polygon_grid(x1_y1[1], x2_y2[1], x1_y1[0], x2_y2[0])
#polygon_grid(y_lim[0], y_lim[1], x_lim[0], x_lim[1])

def exagon(r,y_lim,x_lim):

    #r = 500  # for the hexagon size

    #y_lim = (393500,401000)                                    # y coordinates (boundaries of city of Manchester)
    #x_lim = (382500,389500)                                    # x coordinates (boundaries of city of Manchester)
    xmin =x_lim[0]
    xmax =x_lim[1]
    ymin =y_lim[0]
    ymax =y_lim[1]



    # twice the height of a hexagon's equilateral triangle
    h = int(r * math.sqrt(3))

    polygons = []
    
    # create the hexagons
    for x in range(xmin, xmax, h):
        k=1
        for y in range(ymin, ymax, int(h * h / r / 2)):
            if k==0:
                x=x+r * math.sqrt(3)/2
                hexagon = shape(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y + r],
                                [x + h / 2, y + r / 2],
                                [x + h / 2, y - r / 2],
                                [x, y - r],
                                [x - h / 2, y - r / 2],
                                [x - h / 2, y + r / 2],
                                [x, y + r],
                            ]
                        ],
                    }
                )
                polygons.append(hexagon)
                x=x-r * math.sqrt(3)/2
                k=1
            elif k==1:
                hexagon = shape(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y + r],
                                [x + h / 2, y + r / 2],
                                [x + h / 2, y - r / 2],
                                [x, y - r],
                                [x - h / 2, y - r / 2],
                                [x - h / 2, y + r / 2],
                                [x, y + r],
                            ]
                        ],
                    }
                )
                polygons.append(hexagon)
                k=0
                    

    poly_grid = gpd.GeoDataFrame({'geometry': polygons})
    poly_grid.plot(ax=base, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
    poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
    print('ciao')


#exagon grid
exagon(350,y_lim,x_lim)

traffic_points = gpd.read_file(os.getcwd()+'\\shapefiles\\traffic_points.shp')
print(traffic_points)

polys = gpd.read_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')

points_polys = gpd.sjoin(traffic_points, polys, how="right")
print(points_polys.head())
print(points_polys.info())
print(points_polys['index_left'].unique())
# print(points_polys.loc[points_polys.index_left == 0, 'cars_and_t'].count())

# Calculate the average of the traffic counts in each grid unit
stats_pt = points_polys.groupby('index_left')['cars_and_t'].agg(['mean'])
stats_pt.columns = ["_".join(x) for x in stats_pt.columns.ravel()]
print(stats_pt)
plt.show()


