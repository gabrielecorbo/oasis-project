'''
This file creates the plots of the data and a shapefile from the optimimal point.
'''
import pandas as pd
import importlib
import optimisation_exa as opte   #scripts.
importlib.reload(opte)
import os
from shapely.geometry import Point
import geopandas as gpd

# Import GIS data and car park location data
GIS_data = pd.read_csv(os.getcwd()+'\\dati.csv')
GIS_df = pd.DataFrame(GIS_data)
car_park_data = GIS_df.iloc[:,[0,3,4]]
car_park_df = pd.DataFrame(car_park_data)

opte.gen_sets(GIS_df,car_park_df)
opte.gen_parameters(GIS_df,car_park_df)
opte.gen_demand(GIS_df)
opt_loc,stat=opte.optimize(GIS_df,car_park_df)


opt_loc=car_park_df.iloc[opt_loc]
print(opt_loc[['centroid_x', 'centroid_y']])

def point_df_to_gdf2(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['centroid_x', 'centroid_y']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df
    
    
opt_loc_gdf = point_df_to_gdf2(opt_loc)
opt_loc_gdf.to_file(os.getcwd()+'\\shapefiles\\opt_loc_exa.shp')

#print(opt_loc)