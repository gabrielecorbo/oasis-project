# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import shapefile as shp
import seaborn as sns
import math
from scipy.spatial import distance
from shapely.geometry import Point,LineString,Polygon
from pulp import *
import importlib
import scripts.neighbors as neigh #scripts.
importlib.reload(neigh)
from import_data import exagon,read_shapefile
from datetime import datetime
# %%
# Import GIS data and car park location data
GIS_data = pd.read_csv(os.getcwd()+'\\dati.csv')
GIS_df = pd.DataFrame(GIS_data)
car_park_data = GIS_df.iloc[:,[0,3,4]]
car_park_df = pd.DataFrame(car_park_data)
# %%
y_lim = (51.46,51.56)                                    # y coordinates 
x_lim = (-0.21,0.04)                                    # x coordinates   
# %%
shp_path_roads_1 = os.getcwd()+'\\shapefiles\\gis_osm_roads_free_1.shp'
sf_roads_1 = shp.Reader(shp_path_roads_1)
df_roads = read_shapefile(sf_roads_1)
df_roads['coords'] = df_roads['coords'].apply(LineString)
df_roads = gpd.GeoDataFrame(df_roads, geometry='coords')
rect=Polygon([(x_lim[0],y_lim[0]),(x_lim[0],y_lim[1]),(x_lim[1],y_lim[1]),(x_lim[1],y_lim[0]),(x_lim[0],y_lim[0])])
gdf_roads_clip=df_roads.clip(rect)

#gdf_roads_clip=shp.Reader(os.getcwd()+'\shapefiles\map.shp')
#gdf_roads_clip=read_shapefile(gdf_roads_clip)
#poly_grid=shp.Reader(os.getcwd()+'\shapefiles\grid_exa.shp')
# %%
#import vector of weigthd POIs
data_path = os.getcwd()+'\\poi_df.csv'
data_poi = pd.read_csv(data_path)
poi_gdf = pd.DataFrame(data=data_poi)
data_path = os.getcwd()+'\\vec_poi.csv'
data_poi = pd.read_csv(data_path)
vec_poi = pd.DataFrame(data=data_poi).iloc[:,1]
# %%
#import vector of flow traffic
data_path = os.getcwd()+'\\flow_traffic.csv'
flow_traffic = pd.read_csv(data_path)
flow_traffic = pd.DataFrame(data=flow_traffic).iloc[:,1]
# %%
radius=0.009
polygons,rows,cols,colore,tot_traffic = exagon(radius,y_lim,x_lim)
poly_grid = gpd.GeoDataFrame({'geometry': polygons}) 
# %%
# parameter for sizing
pen_rate=0.04 #penetration rate at year 2025
ut_rate=0.04 #utilization rate (definied as the number of EV that stops for a charge) !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE
wrk_hours=12 #working hours !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE
crg_rate=150 #charging rate in kW
avg_bat=50 #average battery capacity of a EV in kWh 
avg_crg=0.5 #average charging percentage of each session
avg_cap=avg_bat*avg_crg #average charging capacity needed in a charging session
#%%
def gen_sets(df_demand, df_parking):
    """Generate sets to use in the optimization problem"""
    # set of charging demand locations (destinations)
    demand_lc = df_demand.index.tolist()
    # set of candidates for charging station locations (currently existing parking lots)
    chg_lc = df_parking.index.tolist()
    return demand_lc, chg_lc

#%%
def gen_parameters(df_demand, df_parking):
    """Generate parameters to use in the optimization problem,
    including cost to install charging stations, operating costs and others..."""

    N = 6                               # Where vi is the charging possibility of an EV in cell i
    fi = df_demand["Traffico"]          # Where fi is the average traffic in grid i
    di = fi                      # Where di represents the charging demand of EV in grid i
    fti = flow_traffic             # Where fti is the flow traffic in grid i
    #di = di.to_dict()

    # distance matrix of charging station location candidates and charging demand location
    coords_parking = [(x, y) for x, y in zip(df_parking['centroid_x'], df_parking['centroid_y'])]

    coords_demand = [(x, y) for x, y in zip(df_demand['centroid_x'], df_demand['centroid_y'])]

    distance_matrix = distance.cdist(coords_parking, coords_demand, 'euclidean')
    scaling_ratio = 1
    distance_matrix2 = scaling_ratio * distance_matrix
    distance_matrix3 = pd.DataFrame(distance_matrix2, index=df_parking.index.tolist(),
                                    columns=df_demand.index.tolist())
                                    
    #poi_df = qge.read_shapefile(gp_poi)
    coords_pois = [(x, y) for x, y in zip(poi_gdf['long'], poi_gdf['lat'])]
    distance_matrix_poi = distance.cdist(coords_parking, coords_pois, 'euclidean')
    distance_matrix_poi = pd.DataFrame(distance_matrix_poi, index=df_parking.index.tolist())
    print(distance_matrix_poi)
    print(vec_poi)
    #distance_poi = (distance_matrix_poi * vec_poi).sum()
    distance_poi = np.dot(distance_matrix_poi,vec_poi)
    print('distance_poi')
    print(distance_poi)
    print(len(distance_poi))
    max = np.max(distance_poi)
    min = np.min(distance_poi)
    d_poi_scale = (distance_poi - min)/(max - min) 
    plt.hist(d_poi_scale)
    #plt.show()
    #print(distance_poi.head())
    return di, fti, N, distance_matrix3, d_poi_scale

#%%
def gen_demand(df_demand):
    """generate the current satisfied demand for charging for each cell i"""

    diz = df_demand["no_existing_chg"]/(pen_rate*ut_rate*avg_cap/wrk_hours/crg_rate)  # Number of existing chargers in cell i multiplied with a parameter to calculate the satisfied traffic
    #diz = diz.to_dict()

    return diz

#%%
def optimize(df_demand, df_parking):

    # Import i and j set function
    demand_lc, chg_lc = gen_sets(df_demand, df_parking)

    # Import parameters function
    di, fti, N, distance_matrix, d_poi_scale = gen_parameters(df_demand, df_parking)

    # Import current demand of car park z in cell i
    diz = gen_demand(df_demand)

    # set up the optimization problem
    prob = LpProblem('FacilityLocation', LpMaximize)

    x = LpVariable.dicts("UseLocation", [j for j in chg_lc], 0, 1, LpBinary)

    r = np.full([len(demand_lc), len(chg_lc)], None)

    # Create empty dictionary for the remaining demand in cell i
    zip_iterator = zip(demand_lc, [None]*len(demand_lc))
    dr = dict(zip_iterator)

    #print(di)
    #print(diz)
    # For each cell i subtract the existing number of charging stations from the charging demands in cell i
    for i in demand_lc:
        for j in chg_lc:
            dr[i] = di[i] - diz[i] 
            if dr[i] < 0:       # Can't have negative demand therefore limit minimum demand to zero
                dr[i] = 0

    #print(fti)
    # Objective function
    # The scaled distance from the POI is considered as a multiplication factor
    lam = 1
    prob += lpSum((dr[j]*fti[j]*x[j])*(1-lam*d_poi_scale[j]) for j in chg_lc) #

    # Constraints
    for j in chg_lc:
        print((dr[j]*fti[j])*(1-lam*d_poi_scale[j]))
        print(dr[j])
        print(fti[j])
        print(1-lam*d_poi_scale[j])
        print('___________________________')
        nei_j = neigh.neighbors(rows,cols,j)[0]
        nei_j.append(j)
        prob += lpSum(x[k] for k in nei_j) <= 1                                # Constraint 1
    prob += lpSum(x[j] for j in chg_lc) == N                            # Constraint 2

    prob.solve()
    print("Status: ", LpStatus[prob.status])
    #print([x[j].varValue for j in range(len(x))])
    tolerance = .9
    opt_location = []
    for j in chg_lc:
        if x[j].varValue > tolerance:   # If binary value x is positive then the car park has been selected
            opt_location.append(j)
            print("Establish charging station at parking lot", j)
    df_status = pd.DataFrame({"status": [LpStatus[prob.status]], "Tot_no_chargers": [len(opt_location)]})
    print("Final Optimisation Status:\n", df_status)
    
    #print(len(chg_lc))
    #print(len(demand_lc))

    #print(prob.variables())
    varDic = {}
    for variable in prob.variables():
        var = variable.name
        if var[:5] == 'no_of':      # Filter to obtain only the variable 'no_of_chgrs_station_j'
            varDic[var] = variable.varValue

    #print(varDic)
    for variable in prob.variables():
        var = variable.name
#         print(var)
#         print(variable.varValue)

    var_df = pd.DataFrame.from_dict(varDic, orient='index', columns=['value'])
    # Sort the results numerically
    sorted_df = var_df.index.to_series().str.rsplit('_').str[-1].astype(int).sort_values()
    var_df = var_df.reindex(index=sorted_df.index)
    var_df.reset_index(inplace=True)

    location_df = pd.DataFrame(opt_location, columns=['opt_car_park_id'])
#     print(location_df.head())
#     print(car_park_df.head())
    opt_loc_df = pd.merge(location_df, car_park_df, left_on='opt_car_park_id',  right_index=True, how='left')
    opt_loc_df2 = pd.merge(opt_loc_df, var_df, left_on='opt_car_park_id',  right_index=True, how='left')
#     opt_loc_df2.to_csv(path_or_buf='optimal_locations.csv')
    
    v1tot=[]
    v2tot=[]
    for i in opt_location:
        v=neigh.neighbors(rows,cols,i)
        v1tot = v1tot + v[0]
        v2tot = v2tot + v[1]

    v2tot = [int(x) for x in v2tot] 
    v1tot = [int(x) for x in v1tot] 
    opt_location = [int(x) for x in opt_location] 
    color = pd.DataFrame(['white']*len(df_parking)).transpose()
    color[v2tot] = 'yellow'
    color[v1tot] = 'orange'
    color[opt_location] = 'red'
    pol2=[polygons[i] for i in v2tot]
    poly_grid2 = gpd.GeoDataFrame({'geometry': pol2})
    poly_grid2.to_file(os.getcwd()+'\\shapefiles\\exa_2.shp')
    pol1=[polygons[i] for i in v1tot]
    poly_grid1 = gpd.GeoDataFrame({'geometry': pol1})
    poly_grid1.to_file(os.getcwd()+'\\shapefiles\\exa_1.shp')
    optpol=[polygons[i] for i in opt_location]
    poly_grid_opt = gpd.GeoDataFrame({'geometry': optpol})
    poly_grid_opt.to_file(os.getcwd()+'\\shapefiles\\exa_opt.shp')
    
    print('Done')
    return opt_location, df_status, opt_loc_df, opt_loc_df2, color

# %%
gen_sets(GIS_df,car_park_df)
gen_parameters(GIS_df,car_park_df)
gen_demand(GIS_df)
opt_loc, stat, opt_loc_df, opt_loc_df2, color_opt = optimize(GIS_df,car_park_df)
#print(opt_loc_df)

#%%
base = gdf_roads_clip.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
plot = sns.scatterplot(ax=base, x=opt_loc_df['centroid_x'], y=opt_loc_df['centroid_y'], color='mediumblue', legend='full')
plot.set_xlim(x_lim[0], x_lim[1])
plot.set_ylim(y_lim[0], y_lim[1])
plot.set_title(f'Optimal locations for {len(opt_loc)} chargers')

for line in range(opt_loc_df2.shape[0]):
    plot.text(opt_loc_df2.centroid_x[line], opt_loc_df2.centroid_y[line],
                opt_loc_df2.ID[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
plt.show()


# %%
col_opt = (color_opt).values.tolist()[0]

base = gdf_roads_clip.plot(figsize=(12, 8), color='black', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
base.set_xlim(x_lim)
base.set_ylim(y_lim)    
poly_grid.plot(ax=base, facecolor=col_opt, edgecolor='black', lw=0.5, zorder=15,alpha=0.55)

# %%
opt_locations=car_park_df.iloc[opt_loc]
print(opt_locations[['centroid_x', 'centroid_y']])
traff_opt_locations=GIS_df.iloc[opt_loc]
print(traff_opt_locations[['Traffico', 'no_existing_chg']])
def point_df_to_gdf2(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['centroid_x', 'centroid_y']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df
    
    
opt_loc_gdf = point_df_to_gdf2(opt_locations)
opt_loc_gdf.to_file(os.getcwd()+'\\shapefiles\\opt_loc_exa.shp')

# %%
#sizing 

i=0
num_col=np.zeros(len(opt_loc))
for k in opt_loc:
    num_col[i]=math.ceil(GIS_df['Traffico'][k]*pen_rate*ut_rate*avg_cap/wrk_hours/crg_rate)
    i=i+1

base = gdf_roads_clip.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
plot = sns.scatterplot(ax=base, x=opt_loc_df['centroid_x'], y=opt_loc_df['centroid_y'], color='mediumblue', legend='full')
plot.set_xlim(x_lim[0], x_lim[1])
plot.set_ylim(y_lim[0], y_lim[1])
plot.set_title(f'Optimal locations for {len(opt_loc)} chargers')

for line in range(opt_loc_df2.shape[0]):
    plot.text(opt_loc_df2.centroid_x[line], opt_loc_df2.centroid_y[line],
                num_col[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
plt.show()

# %%
# parameters for economic analyisis connecting to medium voltage MTA3 (media tensione)

deploy_cost= 50000 # deployment cost per each station in € !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE
manag_cost = 25.88 #capex for each pod
power_fee = 58.25 #capex according to max power
distance_fee = 487.19+48.79 # to be check!!

ene_cost= 0.15+0.05+0.01 # cost of electric energy €/kWh + comprensive items + excise
maintenance_cost = 500 # annual cost of maintenance for each col
manag_cost_opex = 1113.24 #opex for each pod
power_fee_opex = 46.25 #opex according to max power

ene_reve= 0.35 # revenue for electric energy €/kWh !!!!!!!!!!!!!17/05 DA CORREGGERE CON LORO VALORE

#economic analysis

i=0
daily_rev=np.zeros(len(opt_loc))
brk=np.zeros(len(opt_loc))
roi=np.zeros(len(opt_loc))
years=3 #years for the roi

CAPEX=np.zeros(len(opt_loc))
OPEX=np.zeros(len(opt_loc))

for k in opt_loc:
    daily_rev[i]=(ene_reve-ene_cost)*GIS_df['Traffico'][k]*pen_rate*ut_rate*avg_cap #revenue calcolate per soddisfare energia richiesta effettiva, non massima erogabile
    CAPEX[i]=(deploy_cost+power_fee*crg_rate)*num_col[i]+manag_cost+distance_fee
    OPEX[i]=manag_cost_opex+(power_fee_opex*crg_rate+maintenance_cost)*num_col[i]
    brk[i]=math.ceil((CAPEX[i]/(daily_rev[i]-OPEX[i]/365)))
    roi[i]=100*((years*365*daily_rev[i])-(CAPEX[i]+OPEX[i]*years))/(CAPEX[i])
    i=i+1
print(brk)
print(roi)
min_brk = min(brk)
min_brk_index = np.where(brk == min_brk)
j=min_brk_index[0].tolist()
print(opt_loc[j[0]])
#print(GIS_df['ID'][opt_loc[j[0]]])
print("The best charging station is at parking lot", opt_loc[j[0]])
print("The breakeven point is at", min_brk , 'days')
print("The expected return of investment after",years,"years is",math.ceil(roi[j]),'%')