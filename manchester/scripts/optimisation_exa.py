import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import *
from scipy.spatial import distance
#from plot_roads import read_shapefile, plot_roads
import math
import importlib
import scripts.neighbors as neigh
importlib.reload(neigh)
#from sklearn.preprocessing import scale,MinMaxScaler

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

# Import GIS data and car park location data
GIS_data = pd.read_csv(os.getcwd()+'\\dati.csv')
GIS_df = pd.DataFrame(GIS_data)
GIS_df['mixed_use_area_per_cell']=GIS_df['mixed_use_area_per_cell'].fillna(0)

car_park_data = GIS_df.iloc[:,[0,4,5]]
car_park_df = pd.DataFrame(car_park_data)

#print(GIS_df)
#print(car_park_df)

def gen_sets(df_demand, df_parking):
    """Generate sets to use in the optimization problem"""
    # set of charging demand locations (destinations)
    demand_lc = df_demand.index.tolist()
    # set of candidates for charging station locations (currently existing parking lots)
    chg_lc = df_parking.index.tolist()
    return demand_lc, chg_lc


def gen_parameters(df_demand, df_parking):
    """Generate parameters to use in the optimization problem,
    including cost to install charging stations, operating costs and others..."""

    v0 = 0.05   # the charging possibility of an EV in cell i
    u = 0.10    # the EV penetration rate (utilisation rate) - 10 % of each day are used for charging
    pe = 0.17   # price of electricity per kWh (£/kWh)
    lj = 10     # maximum number of chargers in a station
    alpha = 52  # Average battery capacity (kWh)
    N = 10      # Total number of stations to be installed
    r= 150      # "Radius" of the exagon
    
    Ai = df_demand["mixed_use_area_per_cell"]  # Ai stands for sum of area of the mixed use parts in cell i
    A = math.sqrt(3)*1.5*r*r             # A is the total area of cell i
    vi = Ai / A * v0                           # Where vi is the charging possibility of an EV in cell i
    fi = df_demand["Traffico"]          # Where fi is the average traffic flow in grid i
    di = u * vi * fi                           # Where di represents the charging demand of EV in grid i
    di = di.to_dict()

    # Fast Chargers
    df_demand['m'] = 2                       # Number of charging sessions per day (session/day)
    m = df_demand['m'].to_dict()
    df_demand['p'] = 2                       # Cost of charging per minute (£/minute) (approx £6-7/30min)
    p = df_demand['p'].to_dict()
    df_demand['t'] = 240                     # Charging time for an EV (minutes)
    t = df_demand['t'].to_dict()
    df_demand['ci_j'] = 1000                 # Installation cost
    ci_j = df_demand['ci_j'].to_dict()
    df_demand['cr_j'] = 30                   # cr_j represents the parking fee per day of parking lot j
    cr_j = df_demand['cr_j'].to_dict()
    df_demand['ce_j'] = 1100                 # ce_j represents the price of a charger in station j
    ce_j = df_demand['ce_j'].to_dict()

    # distance matrix of charging station location candidates and charging demand location
    coords_parking = [(x, y) for x, y in zip(df_parking['centroid_x'], df_parking['centroid_y'])]

    coords_demand = [(x, y) for x, y in zip(df_demand['centroid_x'], df_demand['centroid_y'])]

    distance_matrix = distance.cdist(coords_parking, coords_demand, 'euclidean')
    scaling_ratio = 1
    distance_matrix2 = scaling_ratio * distance_matrix
    distance_matrix3 = pd.DataFrame(distance_matrix2, index=df_parking.index.tolist(),
                                    columns=df_demand.index.tolist())
                                    
    #poi_df = read_shapefile(gp_poi)
    coords_pois = [(x, y) for x, y in zip(poi_df['easting'], poi_df['northing'])]
    distance_matrix_poi = distance.cdist(coords_parking, coords_pois, 'euclidean')
    distance_matrix_poi = pd.DataFrame(distance_matrix_poi, index=df_parking.index.tolist())
    distance_poi = distance_matrix_poi.sum()
    max = np.max(distance_poi)
    min = np.min(distance_poi)
    d_poi_scale = (distance_poi - min)/(max - min) 
    plt.hist(d_poi_scale)
    plt.show()
    #print(distance_poi.head())
    return di, m, p, t, ci_j, cr_j, ce_j, pe, alpha, lj, N, distance_matrix3, d_poi_scale


def gen_demand(df_demand):
    """generate the current demand for charging for each cell i"""

    diz = df_demand["no_existing_chg"]  # Number of existing chargers in cell i
    diz = diz.to_dict()

    return diz


def optimize(df_demand, df_parking):

    # Import i and j set function
    demand_lc, chg_lc = gen_sets(df_demand, df_parking)

    # Import parameters function
    di, m, p, t, ci_j, cr_j, ce_j, pe, alpha, lj, N, distance_matrix, d_poi_scale = gen_parameters(df_demand, df_parking)

    # Import current demand of car park z in cell i
    diz = gen_demand(df_demand)

    # set up the optimization problem
    prob = LpProblem('FacilityLocation', LpMaximize)

    n = LpVariable.dicts("no_of_chgrs_station_j",
                         [j for j in chg_lc],
                         0, lj, LpInteger)
    q = LpVariable.dicts("Remaining_dem_station_j",
                         [j for j in chg_lc],
                         0)
    c = LpVariable.dicts("Tot_costs_station_j",
                         [j for j in chg_lc],
                         0)
    x = LpVariable.dicts("UseLocation", [j for j in chg_lc], 0, 3, LpInteger)

    r = np.full([len(demand_lc), len(chg_lc)], None)

    for i in demand_lc:
        for j in chg_lc:
            if distance_matrix[i][j] <= 500:
                r[i][j] = 1
            else:
                r[i][j] = 0
    count = np.count_nonzero(r == 1)
    #print("The number of potential connections with a distance less than 500m is:", count)

    # Objective function
    # The scaled distance from the POI is considered as a multiplication factor
    prob += lpSum((p[j] * t[j] * q[j] - c[j])*(1-d_poi_scale[j]) for j in chg_lc)

    # Create empty dictionary for the remaining demand in cell i
    zip_iterator = zip(demand_lc, [None]*len(demand_lc))
    dr = dict(zip_iterator)

    # For each cell i subtract the existing number of charging stations from the charging demands in cell i
    for i in demand_lc:
        for j in chg_lc:
            dr[i] = di[i] - diz[i] * m[j]
            if dr[i] < 0:       # Can't have negative demand therefore limit minimum demand to zero
                dr[i] = 0

    # Constraints
    for j in chg_lc:
        prob += c[j] == (cr_j[j] + ce_j[j] + ci_j[j] + 0.1 * ce_j[j] + 0.1 * ci_j[j]) * n[j] \
                + pe * alpha * q[j]
    for j in chg_lc:
        prob += q[j] - n[j] * m[j] <= 0                                 # Constraint 1
    for j in chg_lc:
        prob += q[j] - lpSum(r[i][j] * dr[i] for i in demand_lc) <= 0   # Constraint 2
    for i in chg_lc:
        prob += lpSum(x[j] * r[i][j] for j in chg_lc) - 1 <= 0          # Constraint 3
    for j in chg_lc:
        prob += n[j] - x[j] >= 0                                        # Constraint 4
    for j in chg_lc:
        prob += n[j] - lj * x[j] <= 0                                   # Constraint 5

    prob += lpSum(x[j] for j in chg_lc) == N                            # Constraint 6

    prob.solve()
    print("Status: ", LpStatus[prob.status])
    tolerance = .00001
    opt_location = []
    for j in chg_lc:
        if x[j].varValue > tolerance:   # If binary value x is positive then the car park has been selected
            opt_location.append(j)
            print("Establish charging station at parking lot", j)
    df_status = pd.DataFrame({"status": [LpStatus[prob.status]], "Tot_no_chargers": [len(opt_location)]})
    print("Final Optimisation Status:\n", df_status)

    varDic = {}
    for variable in prob.variables():
        var = variable.name
        if var[:5] == 'no_of':      # Filter to obtain only the variable 'no_of_chgrs_station_j'
            varDic[var] = variable.varValue

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

    # Import the road shapefiles
    shp_path_roads_1 = os.getcwd()+'\\shapefiles\\SD_region.shp'
    shp_path_roads_2 = os.getcwd()+'\\shapefiles\\SJ_region.shp'
    sf_roads_1, sf_roads_2 = (shp.Reader(shp_path_roads_1), shp.Reader(shp_path_roads_2, encoding='windows-1252'))
    df_roads_1, df_roads_2 = (read_shapefile(sf_roads_1), read_shapefile(sf_roads_2))
    df_roads = pd.concat([df_roads_1, df_roads_2])  # Combine road dataframes into single dataframe

    roads_df = df_roads_exc_mtrwy #accrocco al volo

    base = roads_df.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
    plot = sns.scatterplot(ax=base, x=opt_loc_df['centroid_x'], y=opt_loc_df['centroid_y'], color='dodgerblue', legend='full')
    plot.set_xlim(382181, 389681)
    plot.set_ylim(393634, 402134)
    plot.set_title(f'Optimal locations for {N} chargers')

    for line in range(0, opt_loc_df2.shape[0]):
        plot.text(opt_loc_df2.centroid_x[line] + 50, opt_loc_df2.centroid_y[line],
                  opt_loc_df2.value[line], horizontalalignment='left',
                  size='medium', color='black', weight='semibold')
                  
#    print(opt_loc_df2)
    
    plt.show()
    
    #rows =  34#int((401000-393500)/(2*150)) 
    #cols =  28#int((389500-382500)/(math.sqrt(3)*150))
    v1tot=[]
    v2tot=[]
    #pol1=[]
    #pol2=[]
    #optpol=[]
    for i in opt_location:
        v=neigh.neighbors(rows,cols,i)
        v1tot = v1tot + v[0]
        v2tot = v2tot + v[1]
    
    pol1=[polygons[int(i)] for i in v1tot]
    #for i in range(len(v1tot)):
    #    pol1.append(polygons[v1tot[i]])
    poly_grid1 = gpd.GeoDataFrame({'geometry': pol1})
    poly_grid1.to_file(os.getcwd()+'\\shapefiles\\exa_1.shp')
    pol2=[polygons[int(i)] for i in v2tot]
    #for i in range(len(v2tot)):
    #    pol2.append(polygons[v2tot[i]])
    #pol2=polygons[enumerate(v2tot)]
    poly_grid2 = gpd.GeoDataFrame({'geometry': pol2})
    poly_grid2.to_file(os.getcwd()+'\\shapefiles\\exa_2.shp')
    optpol=[polygons[int(i)] for i in opt_location]
    #for i in range(len(opt_location)):
    #    optpol.append(polygons[opt_location[i]])
    #optpol=polygons[enumerate(opt_location)]
    poly_grid_opt = gpd.GeoDataFrame({'geometry': optpol})
    poly_grid_opt.to_file(os.getcwd()+'\\shapefiles\\exa_opt.shp')
    
    print('Done')
    return opt_location, df_status
