import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import *
from scipy.spatial import distance
#from plot_roads import read_shapefile, plot_roads
import math

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

# Import GIS data and car park location data
GIS_data = pd.read_csv('D:/ricca/Documents/ASP/Github/oasis-project/manchester/csv_files/mean_car_count_per_grid.csv')
car_park_data = pd.read_csv('D:/ricca/Documents/ASP/Github/oasis-project/manchester/csv_files/council_car_parks_in_grid.csv')
existing_chg_data = pd.read_csv('D:/ricca/Documents/ASP/Github/oasis-project/manchester/csv_files/existing_ev_charging_locations_touching.csv')

GIS_df = pd.DataFrame(GIS_data)
car_park_df = pd.DataFrame(car_park_data)
existing_chg_df = pd.DataFrame(existing_chg_data)

# Create demand centroids for each cell i
GIS_df['centroid_x'] = (GIS_df['right'] + GIS_df['left'])/2
GIS_df['centroid_y'] = (GIS_df['top'] + GIS_df['bottom'])/2

# Group by id, if id > 1 then there are more than 1 charger in each cell i
existing_chg_df2 = existing_chg_df.groupby(by=['fid']).count().reset_index()

# Drop unneeded columns
drop_columns = ['left', 'top', 'right', 'bottom', 'id', 'latitude', 'longitude']
existing_chg_df2 = existing_chg_df2.drop(labels=drop_columns, axis=1)

# Merge the demand grids ids 'fid' between the two dataframes
GIS_df2 = pd.merge(GIS_df, existing_chg_df2, how='left', on='fid')
GIS_df['no_existing_chg'] = GIS_df2['latitude_touch']
GIS_df.sort_values('fid', ascending=True)


def gen_sets(df_demand, df_parking):
    """Generate sets to use in the optimization problem"""
    # set of charging demand locations (destinations)
    demand_lc = df_demand.index.tolist()
    # set of candidates for charging station locations (currently existing parking lots)
    chg_lc = df_parking.index.tolist()

    return demand_lc, chg_lc

