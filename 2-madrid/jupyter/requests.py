import googlemaps
import pandas as pd
from datetime import datetime
gmaps = googlemaps.Client(key = 'AIzaSyAXSd2GOBZnJTRu8_i9RDUVmd_adQeGLM4') #key='Add Your Key here'

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
GIS_df['New_traffic'] = traffic

#%%
'''''
N.B. i valori delle percentuali son cambiati da prima
'''''
new_col = []
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