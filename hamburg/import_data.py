# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib notebook
import seaborn as sns
import shapefile as shp
import pyproj
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
import mapclassify as mc
from scipy.spatial import distance
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon
import csv
import os
import importlib
from pulp import *
import math
import scripts.neighbors as neigh #scripts.
importlib.reload(neigh)
import copy
# %%
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)
# %%
#locate Hamburg traffic data
data_path = os.getcwd()+'\\csv_files\\traffic_points.csv'
data = pd.read_csv(data_path)
raw_traffic_df = pd.DataFrame(data=data)
# %%
#load poi
gp_poi=shp.Reader(os.getcwd()+'\shapefiles\gis_osm_pois_free_1.shp')
# %%
def point_df_to_gdf(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['long', 'lat']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df
    
def polygon_df_to_gdf(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""
    geometry = [box(x1, y1, x2, y2) for x1,y1,x2,y2 in zip(df.left, df.bottom, df.right, df.top)]
    df = df.drop(['left', 'bottom', 'right', 'top'], axis=1)
    geodf = gpd.GeoDataFrame(df, geometry=geometry)
    return df

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
# %%
traffic_points_gdf = point_df_to_gdf(raw_traffic_df)
# %%
traffic_points_gdf.to_file(os.getcwd()+'\\shapefiles\\traffic_points.shp')
# %%
#shp_charg_stat = shp.Reader(os.getcwd()+'\\shapefiles\\charging_stations.shp')
shp_charg_stat = shp.Reader(os.getcwd()+'\\shapefiles\\EV_points.shp')
#print(shp_charg_stat.records())
# %%
shp_path_roads_1 = os.getcwd()+'\\shapefiles\\gis_osm_roads_free_1.shp'
sf_roads_1 = shp.Reader(shp_path_roads_1)
df_roads = read_shapefile(sf_roads_1)
df_roads['coords'] = df_roads['coords'].apply(LineString)
df_roads = gpd.GeoDataFrame(df_roads, geometry='coords')
# %%
df_charg_stat = read_shapefile(shp_charg_stat)
df_charg_stat['long']=[df_charg_stat['coords'][i][0][0] for i in range(len(df_charg_stat))]
df_charg_stat['lat']=[df_charg_stat['coords'][i][0][1] for i in range(len(df_charg_stat))]
drop_columns = ['coords']
df_charg_stat = df_charg_stat.drop(labels=drop_columns, axis=1)
# %%
existing_chargers_gdf = point_df_to_gdf(df_charg_stat)
existing_chargers_gdf.to_file(os.getcwd()+'\\shapefiles\\existing_chargers.shp')
# %%
y_lim = (53.54,53.58)                                    # y coordinates (boundaries of city of Hamburg)
x_lim = (9.94,10.03)                                    # x coordinates (boundaries of city of Hamburg)                                  
# %%
rect=Polygon([(x_lim[0],y_lim[0]),(x_lim[0],y_lim[1]),(x_lim[1],y_lim[1]),(x_lim[1],y_lim[0]),(x_lim[0],y_lim[0])])
rect_gdf=gpd.GeoDataFrame([1], geometry = [rect], crs=27700)
# %%
# Convert poi
poi_df = read_shapefile(gp_poi)
poi_df['long']=[poi_df['coords'][i][0][0] for i in range(len(poi_df))]
poi_df['lat']=[poi_df['coords'][i][0][1] for i in range(len(poi_df))]
poi_gdf = point_df_to_gdf(poi_df)
poi_gdf=poi_gdf.clip(rect)
#poi_gdf = poi_gdf.drop(labels=['name'], axis=1)
#print(poi_gdf)
#coll=['osm_id','code','fclass','coords','longitud','latitud']
coll=['coords']
for col in coll:
    #print(col)
    poi_gdf[col] = poi_gdf[col].astype(str)
poi_gdf.to_file(os.getcwd()+'\\shapefiles\\pois_clip.shp')
print(poi_gdf)
poi_df_clip=pd.DataFrame()
poi_df_clip['long']=poi_gdf['long']
poi_df_clip['lat']=poi_gdf['lat']
poi_df=pd.DataFrame(poi_df_clip)
poi_df.to_csv(os.getcwd()+'\\poi_df.csv')
# %%
traffic_points_clip=traffic_points_gdf.clip(rect)
traffic_points_clip.to_file(os.getcwd()+'\\shapefiles\\traffic_points_clip.shp')
# %%
gdf_roads_clip=df_roads.clip(rect)
#drop_columns = ['coords']
#gdf_roads_clip = gdf_roads_clip.drop(labels=drop_columns, axis=1)
#gdf_roads_clip.to_file(os.getcwd()+'\\shapefiles\\map.shp')
# %%
base = gdf_roads_clip.plot(figsize=(12, 8), color='deepskyblue', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)
# %%
base = gdf_roads_clip.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)
traffic_points_clip.plot(ax=base,column='flow',legend=True)#, cmap='cool')
# %%
(poi_gdf['fclass'].unique())     #135
# %%
"""#poi_gdf=poi_gdf.clip(rect)
vec_poi=np.ones(len(poi_gdf))
print(poi_gdf['fclass'][2])
for i in range(len(poi_gdf)):
  if poi_gdf["fclass"].iloc[i]=='bench' or poi_gdf["fclass"].iloc[i]=='camera_surveillance' or poi_gdf["fclass"].iloc[i]=='drinking_water':
    vec_poi[i]=0
  if poi_gdf["fclass"].iloc[i]=='hotel' or poi_gdf["fclass"].iloc[i]=='restaurant' or poi_gdf["fclass"].iloc[i]=='cinema':
    vec_poi[i]=2
        
#vec_poi=pd.DataFrame(vec_poi.transpose())
print(vec_poi) """

No_weight = pd.Series(['bench','camera_surveillance','drinking_water','comms_tower','recycling_glass',
                       'recycling','car_wash','waste_basket','recycling_paper','recycling_metal',
                       'water_well','toilet','police','recycling_clothes','vending_machine',
                       'vending_cigarette'])

Food      = pd.Series(['bar', 'convenience', 'greengrocer', 'cafe', 'restaurant',
             'fast_food', 'supermarket', 'bakery', 'butcher', 'market_place', 'biergarten',
             'food_court', 'ice_rink','pub','kiosk','beverages','vending_parking'])

Retail    = pd.Series(['artwork', 'post_box', 'hairdresser',
               'bookshop', 'bicycle_shop', 'furniture_shop', 'toy_shop', 'beauty_shop',
               'general', 'telephone', 'doityourself', 'mobile_phone_shop', 'clothes',
               'sports_shop', 'stationery', 'department_store', 'jeweller', 'video_shop',
               'travel_agent', 'optician', 'shoe_shop','bicycle_rental','laundry',
               'car_dealership','florist','car_rental','computer_shop','vending_any',
               'gift_shop','garden_centre','newsagent'])

Leisure   = pd.Series(['playground', 'park','dog_park', 'cinema',
                'nightclub', 'town_hall', 'swimming_pool', 'mall', 'shelter', 'outdoor_shop',
                'arts_centre',
                'golf_course', 'fire_station', 'courthouse', 'fort', 'chalet', 'nursing_home',
                'theme_park', 'water_tower','community_centre','pitch','attraction','theatre',
                'track'])

Tourism   = pd.Series(['hotel', 'monument', 'fountain', 'post_office', 'memorial',
                'observation_tower', 'tourist_info', 'viewpoint', 'ruins', 'castle',
                'wayside_cross', 'picnic_site', 'museum', 'battlefield',
                'embassy','guesthouse','hostel','archaeological','tower','motel',
                'windmill','water_mill','car_sharing','wayside_shrine'])

Finance   = pd.Series(['bank', 'atm'])

Health    = pd.Series(['pharmacy', 'dentist', 'hospital', 'clinic','sports_centre','veterinary','chemist','doctors'])

Education = pd.Series(['kindergarten', 'school', 'college', 'university', 'library'])





vec_poi=np.ones(len(poi_gdf))
for i in range(len(poi_gdf)):
    if poi_gdf["fclass"].iloc[i] in No_weight.values:
        vec_poi[i]=0 # cluster ignobili che pesano 0
    if poi_gdf["fclass"].iloc[i] in Food.values:
        vec_poi[i]=0.29 # cluster Food
    if poi_gdf["fclass"].iloc[i] in Retail.values:
        vec_poi[i]=0.27 # cluster Retail
    if poi_gdf["fclass"].iloc[i] in Leisure.values:
        vec_poi[i]=0.17 # cluster Leisure
    if poi_gdf["fclass"].iloc[i] in Tourism.values:
        vec_poi[i]=0.14 # cluster Tourism   
    if poi_gdf["fclass"].iloc[i] in Finance.values:
        vec_poi[i]=0.08 # cluster Finance  
    if poi_gdf["fclass"].iloc[i] in Health.values:
        vec_poi[i]=0.02 # cluster Health
    if poi_gdf["fclass"].iloc[i] in Education.values:
        vec_poi[i]=0.02 # cluster Education   
        
vec_poi=pd.DataFrame(vec_poi.transpose())
vec_poi.to_csv(os.getcwd()+'\\vec_poi.csv')
# %%
count_poi = pd.DataFrame(poi_gdf['fclass'].value_counts())
print(count_poi)
# %%
base = gdf_roads_clip.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)
poi_gdf.plot(ax=base,markersize=3,column='fclass') #,color='red'
# %%
base = gdf_roads_clip.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)
existing_chargers_gdf.plot(ax=base,color='green')
# %%
def exagon(r,y_lim,x_lim):
    xmin =x_lim[0]
    xmax =x_lim[1]
    ymin =y_lim[0]
    ymax =y_lim[1]

    # twice the height of a hexagon's equilateral triangle
    h = (r * math.sqrt(3))

    polygons = []
    tot_traffic_pre=[]
    tot_mixed=[]
    tot_chargers=[]
    tot_centroide_x=[]
    tot_centroide_y=[]
    colore=[]
    rows=0
    cols=0
    # create the hexagons
    for x in np.arange(xmin, xmax, h):
        k=1
        for y in np.arange(ymin, ymax, (h * h / r / 2)):
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
            parziale=traffic_points_gdf.clip(hexagon)["flow"].sum()
            tot_traffic_pre.append(parziale)
            #mixed=mean_car_count_gdf.clip(hexagon)["mixed_use_area_per_cell"].mean()
            #tot_mixed.append(mixed)
            chargers=existing_chargers_gdf.clip(hexagon)['long'].count()
            tot_chargers.append(chargers)
            rows+=1
        cols+=1   
    rows=int(rows/cols)
    tot_traffic = copy.copy(tot_traffic_pre)
    for i in range(len(tot_traffic)):
        if tot_traffic_pre[i]==0:
            v_n = neigh.neighbors(rows,cols,i)[0]
            tot_traffic[i] = np.mean([tot_traffic_pre[int(j)] for j in v_n])
    
    mas=max(tot_traffic)
    for k in range(len(tot_traffic)):
        if tot_traffic[k]<=0.0*mas:
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
        csv_writer.writerow(['ID', 'Traffico', 'no_existing_chg', 'centroid_x', 'centroid_y', 'Colore']) #,'mixed_use_area_per_cell'
        for k in range(len(tot_chargers)):
            csv_writer.writerow([k,tot_traffic[k],tot_chargers[k],tot_centroide_x[k],tot_centroide_y[k],colore[k]]) #,tot_mixed[k]
    #poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
    #base = gdf_roads_clip.plot(figsize=(12, 8), color='deepskyblue', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
    #base.set_xlim(x_lim)
    #base.set_ylim(y_lim)    
    #poly_grid.plot(ax=base, facecolor=colore, edgecolor='black', lw=0.5, zorder=15)
    #poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
    return polygons,rows,cols,colore,tot_traffic
# %%
#exagon grid
radius=0.003
polygons,rows,cols,colore,tot_traffic = exagon(radius,y_lim,x_lim)
# %%
poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)    
poly_grid.plot(ax=base, facecolor=colore, edgecolor='black', lw=0.5, zorder=15,alpha=0.55)
#poly_grid.plot(ax=base, facecolor='#999999', edgecolor='black', lw=0.5, zorder=15)
poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
# %%
traffic_points = gpd.read_file(os.getcwd()+'\\shapefiles\\traffic_points.shp')
polys = gpd.read_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
points_polys = gpd.sjoin(traffic_points, polys, how="right")
# %%
# parameter for sizing
pen_rate=0.04 #penetration rate at year 2025
ut_rate=0.04 #utilization rate (definied as the number of EV that stops for a charge) !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE
wrk_hours=12 #working hours !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE
crg_rate=150 #charging rate in kW
avg_bat=50 #average battery capacity of a EV in kWh 
avg_crg=0.5 #average charging percentage of each session
avg_cap=avg_bat*avg_crg #average charging capacity needed in a charging session
# %%
