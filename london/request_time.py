# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pulp import *
import importlib
import scripts.neighbors as neigh #scripts.
importlib.reload(neigh)
from import_data import exagon
from datetime import datetime
# %%
import googlemaps 
gmaps = googlemaps.Client(key = '') #key='Add Your Key here'
#%%
def get_time(origin, destination, dep_time):
    directions_result = gmaps.directions(origin,
                                         destination, 
                                         mode="driving", 
                                         departure_time=dep_time,  
                                         traffic_model="best_guess")
    return destination, directions_result[0]['legs'][0]['duration_in_traffic']['value'], directions_result[0]['legs'][0]['duration']['value']

#%%
# Import GIS data and car park location data
GIS_data = pd.read_csv(os.getcwd()+'\\dati.csv')
GIS_df = pd.DataFrame(GIS_data)
# %%
radius=0.009
y_lim = (51.46,51.56)                                    # y coordinates 
x_lim = (-0.21,0.04)                                    # x coordinates   
polygons,rows,cols,colore,tot_traffic = exagon(radius,y_lim,x_lim)
#%%
l=len(GIS_df)
print(l)
#%%
def centroid_s(i):
    return str(GIS_df.loc[i,'centroid_y'])+', '+str(GIS_df.loc[i,'centroid_x'])

#%%
timetables = []
coords_c = [centroid_s(i) for i in range(l)]
for i in range(l):
    centroids_a = []
    centroids_b = []
    duration_traffic = []
    duration_no_traffic = []
    #neighbours_indexes = [(i, y) for y in neigh.neighbors(rows,cols,i)]
    #neighbours_coords = [centroid, centroid_s(y) for y in neigh.neighbours(rows,cols,i)]
    dep_time = datetime.now() # set times
    origin = i 
    for destination in neigh.neighbors(rows,cols,i)[0]:
        destination=int(destination)
        #print(origin,destination,centroid_s(origin), centroid_s(destination))
        c, dt, dnt = get_time(centroid_s(origin), centroid_s(destination), dep_time)
        centroids_a.append(origin)
        centroids_b.append(destination)
        duration_traffic.append(dt)
        duration_no_traffic.append(dnt)

    timetable = pd.DataFrame([centroids_a,centroids_b,duration_traffic,duration_no_traffic]).transpose()
    timetable.columns = ("Origin","Destination", "Duration_traffic", "Duration_no_traffic")
    timetable
    timetables.append(timetable)


#%%
print(timetables)

#%%
times = np.zeros((l,l))
traffic_times = np.zeros((l,l))
for i in range(l):
    for j in range(len(timetables[i]["Destination"])):
        y = timetables[i].loc[j,"Destination"]
        traffic_times[i,y]=timetables[i].loc[j,"Duration_traffic"]
        times[i,y]=timetables[i].loc[j,"Duration_no_traffic"]

#%%

print(times)
print(traffic_times)

#%%

traff_ex = np.true_divide(traffic_times.sum(1),(traffic_times!=0).sum(1))
traff_en = np.true_divide(traffic_times.sum(0),(traffic_times!=0).sum(0))
time_ex = np.true_divide(times.sum(1),(times!=0).sum(1))
time_en = np.true_divide(times.sum(0),(times!=0).sum(0))

#%%
exiting = traff_ex - time_ex
entering = traff_en - time_en

#%%
print((exiting<0).sum())
print((entering<0).sum())
print((exiting==0).sum())
print((entering==0).sum())

#%%
traffic = entering - exiting
print((traffic==0).sum())
print((traffic<0).sum())
traffic += abs(traffic.min())

#%%
max = np.max(traffic)
min = np.min(traffic)
traffic = (traffic - min)/(max - min) 
plt.hist(traffic) 
#GIS_df['New_traffic'] = traffic
flow_traffic=pd.DataFrame(traffic.transpose())
flow_traffic.to_csv(os.getcwd()+'\\flow_traffic.csv')
#%%
'''''
N.B. i valori delle percentuali son cambiati da prima
'''''
gdf_roads_clip = gpd.read_file(os.getcwd()+'\\shapefiles\\map.shp')
new_col = []
traffic = traffic
mas=np.max(traffic)
for k in range(len(traffic)):
    if traffic[k]<=0.0*mas:
        col='lightcyan'
    elif traffic[k]<=0.25*mas:
        col='lightskyblue'
    elif traffic[k]<=0.5*mas:
        col='deepskyblue'
    elif traffic[k]<=0.75*mas:
        col='royalblue'
    elif traffic[k]<=mas:
        col='darkblue'
    new_col.append(col)
poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with  
base.set_xlim(x_lim)
base.set_ylim(y_lim)    
poly_grid.plot(ax=base, facecolor=new_col, edgecolor='black', lw=0.5, zorder=15,alpha=0.55)
# %%
poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0, alpha=0.5)  # Zorder controls the layering of the charts with  
base.set_xlim(x_lim)
base.set_ylim(y_lim) 
poly_grid.plot(ax=base,facecolor='None', edgecolor='black', lw=0.8, zorder=15, alpha=1)
# %%
# APPROCCIO RAPPORTO ADIMENSIONALE
time_rate_matrix = traffic_times/times
time_rate_matrix = np.nan_to_num(time_rate_matrix)
# %%
traff_ex_rate = np.true_divide(time_rate_matrix.sum(1),(time_rate_matrix!=0).sum(1))
traff_en_rate = np.true_divide(time_rate_matrix.sum(0),(time_rate_matrix!=0).sum(0))
# %%
traffic_rate = traff_ex_rate - traff_en_rate
print((traffic_rate==0).sum())
print((traffic_rate<0).sum())
plt.hist(traffic_rate) 
# %%
max = np.max(traffic_rate)
min = np.min(traffic_rate)
traffic_rate = (traffic_rate - min)/(max - min) 
plt.hist(traffic_rate) 
# %%
gdf_roads_clip = gpd.read_file(os.getcwd()+'\\shapefiles\\map.shp')
new_col = []
mas=np.max(traffic_rate)
for k in range(len(traffic_rate)):
    if traffic_rate[k]<=0.0*mas:
        col='lightcyan'
    elif traffic_rate[k]<=0.25*mas:
        col='lightskyblue'
    elif traffic_rate[k]<=0.5*mas:
        col='deepskyblue'
    elif traffic_rate[k]<=0.75*mas:
        col='royalblue'
    elif traffic_rate[k]<=mas:
        col='darkblue'
    new_col.append(col)
poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with  
base.set_xlim(x_lim)
base.set_ylim(y_lim)    
poly_grid.plot(ax=base, facecolor=new_col, edgecolor='black', lw=0.5, zorder=15,alpha=0.55)
# %%
