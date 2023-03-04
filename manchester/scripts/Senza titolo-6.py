y_lim = (394500, 400500)                                    # y coordinates (boundaries of city of Manchester)
x_lim = (382000, 387500) 

#x1_y1 = (-2.272481011086273, 53.44695395956606)             # latitudes (boundaries of city of Manchester)
#x2_y2 = (-2.1899126640044337, 53.50104467521926)   
#
y_min = y_lim[0]
y_max = y_lim[1] 
x_min = x_lim[0]
x_max = x_lim[1]
"""This function takes the coordinate limits and creates a regular grid
across the area"""

step_size = 500     # Distance in meters
gridpoints = []

x = x_min
while x <= x_max:
    y = y_min
    while y <= y_max:
        p = (x, y)
        gridpoints.append(p)
        y += step_size
    x += step_size

grid_df = pd.DataFrame(data=gridpoints, columns=['x', 'y'])
plt.scatter(grid_df['x'], grid_df['y'], color='maroon', s=2)
# open the file in the write mode
# with open('/optimise_EV_location/gridpoints.csv', 'w') as csv_file:
#     # create the csv writer
#     csv_file.write('hor;vert\n')
#     for p in gridpoints:
#         csv_file.write('{:f};{:f}\n'.format(p.x, p.y))
##open(os.getcwd()+'\\csv_files\\gridpoints.csv', 'w')
#gridpoints.write('hor;vert\n')
plt.show()