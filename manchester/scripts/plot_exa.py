'''
This file creates the plots of the data and a shapefile from the optimimal point.
'''
gen_sets(GIS_df,car_park_df)
gen_parameters(GIS_df,car_park_df)
gen_demand(GIS_df)
opt_loc,stat=optimize(GIS_df,car_park_df)


opt_loc=car_park_df.iloc[opt_loc]
print(opt_loc)

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