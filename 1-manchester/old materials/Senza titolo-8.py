gen_sets(GIS_df,car_park_df)
gen_parameters(GIS_df,car_park_df)
gen_demand(GIS_df)


df_demand=GIS_df
df_parking=car_park_df

# Import i and j set function
demand_lc, chg_lc = gen_sets(df_demand, df_parking)

# Import parameters function
di, m, p, t, ci_j, cr_j, ce_j, pe, alpha, lj, N, distance_matrix = gen_parameters(df_demand, df_parking)

# Import current demand of car park z in cell i
diz = gen_demand(df_demand)

# set up the optimization problem
prob = LpProblem('FacilityLocation', LpMaximize)

n = LpVariable.dicts("no_of_chgrs_station_j",[j for j in chg_lc],0, lj, LpInteger)
q = LpVariable.dicts("Remaining_dem_station_j",[j for j in chg_lc],0)
c = LpVariable.dicts("Tot_costs_station_j",[j for j in chg_lc],0)
x = LpVariable.dicts("UseLocation", [j for j in chg_lc], 0, 1, LpBinary)

r = np.full([len(demand_lc), len(chg_lc)], None)

for i in demand_lc:
    for j in chg_lc:
        if distance_matrix[i][j] <= 500:
            r[i][j] = 1
        else:
            r[i][j] = 0
count = np.count_nonzero(r == 1)
print("The number of potential connections with a distance less than 500m is:", count)

# Objective function
prob += lpSum(p[j] * t[j] * q[j] - c[j] for j in chg_lc)

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

roads_df = df_roads_exc_mtrwy #accrocco

base = roads_df.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
plot = sns.scatterplot(ax=base, x=opt_loc_df['Easting'], y=opt_loc_df['Northing'], color='dodgerblue', legend='full')
plot.set_xlim(382181, 389681)
plot.set_ylim(393634, 402134)
plot.set_title(f'Optimal locations for {N} chargers')

for line in range(0, opt_loc_df2.shape[0]):
    plot.text(opt_loc_df2.Easting[line] + 50, opt_loc_df2.Northing[line],
              opt_loc_df2.value[line], horizontalalignment='left',
              size='medium', color='black', weight='semibold')
              
#    print(opt_loc_df2)

plt.show()

print(opt_location)
print(df_status)
