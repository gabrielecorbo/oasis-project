'''
This file creates the plots of the data and a shapefile from the optimimal point.
'''
gen_sets(GIS_df,car_park_df)
gen_parameters(GIS_df,car_park_df)
gen_demand(GIS_df)
opt_loc,stat=optimize(GIS_df,car_park_df)


#print(opt_loc)
data_path = os.getcwd()+'\\csv_files\\council_car_parks_in_grid.csv'
data = pd.read_csv(data_path)
parking_slot = pd.DataFrame(data=data)

opt_loc=parking_slot.iloc[opt_loc]
opt_loc=opt_loc[["Easting","Northing"]]
print(opt_loc)

def point_df_to_gdf(df):
    """takes a dataframe with columns named 'longitude' and 'latitude'
    to transform to a geodataframe with point features"""

    df['coordinates'] = df[['Easting', 'Northing']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = gpd.GeoDataFrame(df, geometry='coordinates')
    return df
    
    
opt_loc_gdf = point_df_to_gdf(opt_loc)
opt_loc_gdf.to_file(os.getcwd()+'\\shapefiles\\opt_loc.shp')

#print(opt_loc)