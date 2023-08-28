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
from shapely.geometry import shape,Point,LineString,Polygon
from pulp import *
import csv
import copy
# %%
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
def point_df_to_gdf_man(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['longitude', 'latitude']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df
# %%
def neighbors(rows,colums,index):
    mat=np.zeros((rows,colums))
    k=0
    for j in range(colums):
        for i in range(rows)[::-1]:
            mat[i,j]=k
            k+=1
    #print(mat)
    i_inx=rows-index%rows-1
    j_inx=int(index/rows)
    #print(i_inx,j_inx)
    v=[]
    if (i_inx-1)>=0:
        v.append(mat[i_inx-1,j_inx])
    if (i_inx+1)<rows:
        v.append(mat[i_inx+1,j_inx])
    if (j_inx-1)>=0:
        v.append(mat[i_inx,j_inx-1])
    if (j_inx+1)<colums:
        v.append(mat[i_inx,j_inx+1])
    if (rows-i_inx)%2==0:   #seconda quarta sesta ecc riga dal basso
        if (i_inx+1)<rows and (j_inx+1)<colums:
            v.append(mat[i_inx+1,j_inx+1])
        if (i_inx-1)>=0 and (j_inx+1)<colums:
            v.append(mat[i_inx-1,j_inx+1])
    else:
        if (i_inx+1)<rows and (j_inx-1)>=0:
            v.append(mat[i_inx+1,j_inx-1])
        if (i_inx-1)>=0 and (j_inx-1)>=0:
            v.append(mat[i_inx-1,j_inx-1])
    #print(v)

    ##### second level
    v2=[]
    if (i_inx-2)>=0:
        v2.append(mat[i_inx-2,j_inx])
    if (i_inx+2)<rows:
        v2.append(mat[i_inx+2,j_inx])
    if (j_inx-2)>=0:
        v2.append(mat[i_inx,j_inx-2])
    if (j_inx+2)<colums:
        v2.append(mat[i_inx,j_inx+2])
    if (i_inx+2)<rows and (j_inx+1)<colums:
        v2.append(mat[i_inx+2,j_inx+1])
    if (i_inx+2)<rows and (j_inx-1)>=0:
        v2.append(mat[i_inx+2,j_inx-1])
    if (i_inx-2)>=0 and (j_inx+1)<colums:
        v2.append(mat[i_inx-2,j_inx+1])
    if (i_inx-2)>=0 and (j_inx-1)>=0:
        v2.append(mat[i_inx-2,j_inx-1])
    if (rows-i_inx)%2==0:   #seconda quarta sesta ecc riga dal basso
        if (i_inx+1)<rows and (j_inx-1)>=0:
            v2.append(mat[i_inx+1,j_inx-1])
        if (i_inx-1)>=0 and (j_inx-1)>=0:
            v2.append(mat[i_inx-1,j_inx-1])
        if (i_inx+1)<rows and (j_inx+2)<colums:
            v2.append(mat[i_inx+1,j_inx+2])
        if (i_inx-1)>=0 and (j_inx+2)<colums:
            v2.append(mat[i_inx-1,j_inx+2])
    else:
        if (i_inx+1)<rows and (j_inx-2)>=0:
            v2.append(mat[i_inx+1,j_inx-2])
        if (i_inx-1)>=0 and (j_inx-2)>=0:
            v2.append(mat[i_inx-1,j_inx-2])
        if (i_inx+1)<rows and (j_inx+1)<colums:
            v2.append(mat[i_inx+1,j_inx+1])
        if (i_inx-1)>=0 and (j_inx+1)<colums:
            v2.append(mat[i_inx-1,j_inx+1])
    #print(v2)
    return v,v2
# %%
def exagon_man(r,y_lim,x_lim,traffic_points_gdf):
    xmin =x_lim[0]
    xmax =x_lim[1]
    ymin =y_lim[0]
    ymax =y_lim[1]

    # twice the height of a hexagon's equilateral triangle
    h = (r * math.sqrt(3))

    polygons = []
    tot_traffic_pre=[]
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
            parziale=traffic_points_gdf.clip(hexagon)["all_motor_vehicles"].sum()
            tot_traffic_pre.append(parziale)
            rows+=1
        cols+=1   
    rows=int(rows/cols)
    tot_traffic = copy.copy(tot_traffic_pre)
    for i in range(len(tot_traffic)):
        if tot_traffic_pre[i]==0:
            v_n = neighbors(rows,cols,i)[0]
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
    return polygons,rows,cols,colore,tot_traffic
#%%
def gen_sets(df_demand, df_parking):
    """Generate sets to use in the optimization problem"""
    # set of charging demand locations (destinations)
    demand_lc = df_demand.index.tolist()
    # set of candidates for charging station locations (currently existing parking lots)
    chg_lc = df_parking.index.tolist()
    return demand_lc, chg_lc

#%%
def gen_parameters_man(df_demand, df_parking, flow_traffic, poi_gdf, vec_poi):
    """Generate parameters to use in the optimization problem,
    including cost to install charging stations, operating costs and others..."""

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
    coords_pois = [(x, y) for x, y in zip(poi_gdf['longitude'], poi_gdf['latitude'])]
    distance_matrix_poi = distance.cdist(coords_parking, coords_pois, 'euclidean')
    distance_matrix_poi = pd.DataFrame(distance_matrix_poi, index=df_parking.index.tolist())
    distance_poi = np.dot(distance_matrix_poi,vec_poi)
    max = np.max(distance_poi)
    min = np.min(distance_poi)
    d_poi_scale = (distance_poi - min)/(max - min) 
    plt.hist(d_poi_scale)
    #plt.show()
    #print(distance_poi.head())
    return di, fti, distance_matrix3, d_poi_scale

#%%
def gen_demand(df_demand, pen_rate, catch_rate, avg_cap, wrk_hours, crg_rate):
    """generate the current satisfied demand for charging for each cell i"""

    diz = df_demand["no_existing_chg"]/(pen_rate*catch_rate*avg_cap/wrk_hours/crg_rate)  # Number of existing chargers in cell i multiplied with a parameter to calculate the satisfied traffic
    #diz = diz.to_dict()

    return diz

#%%
def optimize_man(df_demand, df_parking, number, flow_traffic, poi_gdf, vec_poi, pen_rate, catch_rate, avg_cap, wrk_hours, crg_rate, rows, cols, car_park_df):

    # Import i and j set function
    demand_lc, chg_lc = gen_sets(df_demand, df_parking)

    # Import parameters function
    di, fti, distance_matrix, d_poi_scale = gen_parameters_man(df_demand, df_parking, flow_traffic, poi_gdf, vec_poi)

    N = number

    # Import current demand of car park z in cell i
    diz = gen_demand(df_demand, pen_rate, catch_rate, avg_cap, wrk_hours, crg_rate)

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
        nei_j = neighbors(rows,cols,j)[0]
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
        v=neighbors(rows,cols,i)
        v1tot = v1tot + v[0]
        v2tot = v2tot + v[1]

    v2tot = [int(x) for x in v2tot] 
    v1tot = [int(x) for x in v1tot] 
    opt_location = [int(x) for x in opt_location] 
    color = pd.DataFrame(['white']*len(df_parking)).transpose()
    color[v2tot] = 'yellow'
    color[v1tot] = 'orange'
    color[opt_location] = 'red'
    
    print('Done')
    return opt_location, df_status, opt_loc_df, opt_loc_df2, color

def manchester_function(N):
    
    local_path = '\\manchester'

    GIS_data = pd.read_csv(os.getcwd()+local_path+'\\dati.csv')
    GIS_df = pd.DataFrame(GIS_data)
    car_park_data = GIS_df.iloc[:,[0,3,4]]
    car_park_df = pd.DataFrame(car_park_data)
    x_lim = (-2.27002, -2.14955)
    y_lim = (53.43895, 53.51545)
    
    #locate madrid traffic data
    data_path = os.getcwd()+local_path+'\\csv_files\\traffic_data.csv'
    data = pd.read_csv(data_path)
    raw_traffic_df = pd.DataFrame(data=data)
    traffic_points_gdf = point_df_to_gdf_man(raw_traffic_df)
    #import vector of weigthd POIs
    data_path = os.getcwd()+local_path+'\\poi_df.csv'
    data_poi = pd.read_csv(data_path)
    poi_gdf = pd.DataFrame(data=data_poi)
    data_path = os.getcwd()+local_path+'\\vec_poi.csv'
    data_poi = pd.read_csv(data_path)
    vec_poi = pd.DataFrame(data=data_poi).iloc[:,1]
    #import vector of flow traffic
    data_path = os.getcwd()+local_path+'\\flow_traffic.csv'
    flow_traffic = pd.read_csv(data_path)
    flow_traffic = pd.DataFrame(data=flow_traffic).iloc[:,1]

    radius=0.006
    
    polygons,rows,cols,colore,tot_traffic = exagon_man(radius,y_lim,x_lim,traffic_points_gdf)
    poly_grid = gpd.GeoDataFrame({'geometry': polygons})
    # parameter for sizing
    pen_rate=0.04 #penetration rate at year 2025
    catch_rate=0.04 #catch rate (definied as the number of EV that stops for a charge) 31/07 stimato, dipende da troppe variabili, spiegare su report..
    wrk_hours=12 #working hours !!!!!!!!!!!!!31/07 loro non danno valore, guardare https://www.mdpi.com/1996-1073/16/6/2619 e https://www.sciencedirect.com/science/article/pii/S258900422201906X
    crg_rate=150 #charging rate in kW 
    avg_bat=50 #average battery capacity of a EV in kWh 
    avg_crg=0.5 #average charging percentage of each session
    avg_cap=avg_bat*avg_crg #average charging capacity needed in a charging session

    number = N
    
    opt_loc, stat, opt_loc_df, opt_loc_df2, color_opt = optimize_man(GIS_df,car_park_df,number, flow_traffic, poi_gdf, vec_poi, pen_rate, catch_rate, avg_cap, wrk_hours, crg_rate, rows, cols, car_park_df)

    
    shp_path_roads_1 = os.getcwd()+local_path+'\\shapefiles\\gis_osm_roads_free_1.shp'
    sf_roads_1 = shp.Reader(shp_path_roads_1)
    df_roads = read_shapefile(sf_roads_1)
    df_roads['coords'] = df_roads['coords'].apply(LineString)
    df_roads = gpd.GeoDataFrame(df_roads, geometry='coords')
    rect=Polygon([(x_lim[0],y_lim[0]),(x_lim[0],y_lim[1]),(x_lim[1],y_lim[1]),(x_lim[1],y_lim[0]),(x_lim[0],y_lim[0])])
    gdf_roads_clip=df_roads.clip(rect)

    col_opt = (color_opt).values.tolist()[0]
    
    return col_opt, poly_grid, x_lim, y_lim, gdf_roads_clip