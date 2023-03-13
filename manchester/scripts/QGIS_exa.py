import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shapefile as shp
import pyproj
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
import mapclassify as mc
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon
import csv
import os


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


#locate manchester traffic data
data_path = os.getcwd()+'\\csv_files\\manchester_traffic_data.csv'
data = pd.read_csv(data_path)
raw_traffic_df = pd.DataFrame(data=data)
#print(raw_traffic_df.info())


#importo mean car count
data_path = os.getcwd()+'\\csv_files\\mean_car_count_per_grid.csv'
data = pd.read_csv(data_path)
mean_car_count = pd.DataFrame(data=data)

#importo Existing EV charging stations
data_path = os.getcwd()+'\\csv_files\\existing_ev_charging_locations_touching.csv'
data = pd.read_csv(data_path)
existing_chargers = pd.DataFrame(data=data)

# Only use traffic data from the year 2019 and drop useless columns
raw_traffic_df = raw_traffic_df[raw_traffic_df['year'] == 2019]
raw_traffic_df = raw_traffic_df.drop(labels=['region_id', 'region_name', 'local_authority_id', 'local_authority_name',
                                             'start_junction_road_name', 'end_junction_road_name', 'link_length_miles',
                                             'estimation_method', 'estimation_method_detailed', 'pedal_cycles', 'two_wheeled_motor_vehicles',
                                             'buses_and_coaches','lgvs', 'hgvs_2_rigid_axle', 'hgvs_3_rigid_axle',
                                             'hgvs_4_or_more_rigid_axle', 'hgvs_3_or_4_articulated_axle', 'hgvs_5_articulated_axle',
                                             'hgvs_6_articulated_axle' ,'all_hgvs','all_motor_vehicles','year','road_name','road_type',
                                             'link_length_km','count_point_id'], axis=1)


# Tolgo i NaN con coordinata centroide e tengo solo longitude e latitude del dataframe
#for i in range(len(existing_chargers)):
#    if (math.isnan(existing_chargers.iloc[i,7])):
#        existing_chargers.iloc[i,7]=(existing_chargers.iloc[i,3]+existing_chargers.iloc[i,5])/2
#        existing_chargers.iloc[i,8]=(existing_chargers.iloc[i,2]+existing_chargers.iloc[i,4])/2
#        
drop_columns = ['id', 'latitude_touch', 'name','fid']
existing_chargers = existing_chargers.drop(labels=drop_columns, axis=1)
#Droppo le righe vuote senza existing chargers
#existing_chargers=existing_chargers.iloc[0:156,:]
existing_chargers.dropna(inplace=True)

# Add a coordinate column to the dataframe and convert to UK EPSG:27700 (meters)
proj = pyproj.Transformer.from_crs(4326, 27700, always_xy=True)
x1, y1 = (existing_chargers['longitude'], existing_chargers['latitude'])
x2, y2 = proj.transform(x1, y1)
x2, y2 = (pd.DataFrame(x2, columns=['easting']), pd.DataFrame(y2, columns=['northing']))
existing_chargers = pd.concat([existing_chargers, x2, y2], axis=1)

drop_columns = ['left', 'top', 'right', 'bottom','latitude','longitude']
existing_chargers = existing_chargers.drop(labels=drop_columns, axis=1)
#print(existing_chargers)

# Create demand centroids for each cell i
mean_car_count['easting'] = (mean_car_count['right'] + mean_car_count['left'])/2
mean_car_count['northing'] = (mean_car_count['top'] + mean_car_count['bottom'])/2

#load poi
gp_poi=gpd.read_file(os.getcwd()+'\shapefiles\gis_osm_pois_free_1.shp')


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
    
def polygon_df_to_gdf(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""
    geometry = [box(x1, y1, x2, y2) for x1,y1,x2,y2 in zip(df.left, df.bottom, df.right, df.top)]
    df = df.drop(['left', 'bottom', 'right', 'top'], axis=1)
    geodf = gpd.GeoDataFrame(df, geometry=geometry)
    #print(geodf.head())
    return df


traffic_points_gdf = point_df_to_gdf(raw_traffic_df)
#traffic_points_gdf = raw_traffic_df

#print(traffic_points_gdf.head())
#traffic_points_gdf = traffic_points_gdf.set_crs(crs="EPSG:4326")
#print('Traffic CRS', '\n', traffic_points_gdf.crs)
traffic_points_gdf.to_file(os.getcwd()+'\\shapefiles\\traffic_points.shp')
#sf_traffic_points

#converto da dataframe a gdf
existing_chargers_gdf = point_df_to_gdf(existing_chargers)
#creo shapefile di existing chargers
existing_chargers_gdf.to_file(os.getcwd()+'\\shapefiles\\existing_chargers.shp')

#converto da dataframe a gdf
mean_car_count_gdf = point_df_to_gdf(mean_car_count)
#mean_car_count_gdf = polygon_df_to_gdf(mean_car_count)
print(mean_car_count_gdf.head())

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
    tot_traffic=[]
    tot_mixed=[]
    tot_chargers=[]
    tot_centroide_x=[]
    tot_centroide_y=[]
    colore=[]
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
                centroide_x=x
                centroide_y=y
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
                centroide_x=x
                centroide_y=y
            tot_centroide_x.append(centroide_x)
            tot_centroide_y.append(centroide_y)
            parziale=traffic_points_gdf.clip(hexagon)["cars_and_taxis"].sum()
            tot_traffic.append(parziale)
            #smallClip = gpd.clip(hexagon, mean_car_count_gdf)
            #smallClip_explode = smallClip['geometry'].explode()
            mixed=mean_car_count_gdf.clip(hexagon)["mixed_use_area_per_cell"].mean()
            tot_mixed.append(mixed)
            chargers=existing_chargers_gdf.clip(hexagon)['easting'].count()
            tot_chargers.append(chargers)
            
    mas=max(tot_traffic)
    for k in range(len(tot_traffic)):
        if tot_traffic[k]==0:
            col='lightcyan'
        elif tot_traffic[k]<=0.15*mas:
            col='lightskyblue'
        elif tot_traffic[k]<=0.4*mas:
            col='deepskyblue'
        elif tot_traffic[k]<=0.6*mas:
            col='royalblue'
        elif tot_traffic[k]<=mas:
            col='darkblue'
        colore.append(col)
    with open('dati.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #scriviamo prima la riga di intestazione
        csv_writer.writerow(['ID', 'Traffico', 'no_existing_chg','mixed_use_area_per_cell', 'centroid_x', 'centroid_y', 'Colore'])
        for k in range(len(tot_chargers)):
            csv_writer.writerow([k,tot_traffic[k],tot_chargers[k],tot_mixed[k],tot_centroide_x[k],tot_centroide_y[k],colore[k]])
    poly_grid = gpd.GeoDataFrame({'geometry': polygons})
    poly_grid.plot(ax=base, facecolor=colore, edgecolor='black', lw=0.5, zorder=15)
    poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
    print('ciao')
    print(tot_chargers)
    print(sum(tot_chargers))
    return polygons

#exagon grid
polygons = exagon(150,y_lim,x_lim)

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


