#base = df_roads_exc_mtrwy.plot(figsize=(12, 8), color='deepskyblue', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
#df_roads_exc_mtrwy.plot(ax=base)

traffic_points_gdf = traffic_points_gdf.drop(labels=['year','road_name','road_type'
                                                     ,'link_length_km','count_point_id',
                                                     'easting','northing','latitude','longitude'], axis=1)
traffic_points_gdf.plot( x='horizontal', y='vertical', c='cars_and_taxis', cmap='viridis', kind='scatter', s=7, zorder=10)
plt.show()
