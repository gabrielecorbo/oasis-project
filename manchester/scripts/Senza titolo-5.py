#x1_y1 = (-2.272481011086273, 53.44695395956606)             # latitudes (boundaries of city of Manchester)
#x2_y2 = (-2.1899126640044337, 53.50104467521926)   
#ymin = x1_y1[1]
#ymax = x2_y2[1] 
#xmin = x1_y1[0]
#xmax = x2_y2[0]

y_lim = (394500, 400500)                                    # y coordinates (boundaries of city of Manchester)
x_lim = (382000, 387500) 

#x1_y1 = (-2.272481011086273, 53.44695395956606)             # latitudes (boundaries of city of Manchester)
#x2_y2 = (-2.1899126640044337, 53.50104467521926)   
#
ymin = y_lim[0]
ymax = y_lim[1] 
xmin = x_lim[0]
xmax = x_lim[1]


"""This function takes the coordinate limits and creates a polygon grid
across the area"""

height = 500
width = 500

cols = list(np.arange(xmin, xmax + width, width))
rows = list(np.arange(ymin, ymax + height, height))

polygons = []
for x in cols[:-1]:
     for y in rows[:-1]:
         polygons.append(Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)]))
poly_grid = gpd.GeoDataFrame({'geometry': polygons})
base = df_roads_exc_mtrwy.plot(figsize=(12, 8), color='deepskyblue', lw=0.4, zorder=0)  # Zorder controls the layering of the charts with
poly_grid.plot(ax=base, facecolor="none", edgecolor='black', lw=0.7, zorder=15)

#base = poly_grid.plot(figsize=(12, 8), facecolor="none", edgecolor='black', lw=0.7, zorder=15)  # Zorder controls the layering of the charts with

poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid.shp')
plt.show()
