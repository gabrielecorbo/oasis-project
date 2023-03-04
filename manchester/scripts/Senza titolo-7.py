shp_path_roads_1 = os.getcwd()+'\\shapefiles\\map.shp'

roads_df = df_roads

base = roads_df.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
plt.show()
