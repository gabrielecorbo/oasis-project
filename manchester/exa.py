import json
import math
from shapely.geometry import shape, Point, Polygon
from pyproj import Proj, transform
import geopandas

def exagon(r,y_lim,x_lim):

    #r = 500  # for the hexagon size

    #y_lim = (393500,401000)                                    # y coordinates (boundaries of city of Manchester)
    #x_lim = (382500,389500)                                    # x coordinates (boundaries of city of Manchester)
    xmin =x_lim[0]
    xmax =x_lim[1]
    ymin =x_lim[1]
    ymax =y_lim[1]




    # create the feature collection - empty geojson object
    feature_collection = {"type": "FeatureCollection", "features": []}



    # twice the height of a hexagon's equilateral triangle
    h = int(r * math.sqrt(3))

    polygons = []
    k=1
    # create the hexagons
    for x in range(xmin, xmax, h):
        for y in range(ymin, ymax, int(h * h / r / 2)):
            if k==0:
                x=x+r * math.sqrt(3)/2
                hexagon = shape(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y + r],
                                [x + h / 2, y + r / 2],
                                [x + h / 2, y - r / 2],
                                [x, y - r],
                                [x - h / 2, y - r / 2],
                                [x - h / 2, y + r / 2],
                                [x, y + r],
                            ]
                        ],
                    }
                )
                polygons.append(hexagon)
                x=x-r * math.sqrt(3)/2
                k=1
            elif k==1:
                hexagon = shape(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y + r],
                                [x + h / 2, y + r / 2],
                                [x + h / 2, y - r / 2],
                                [x, y - r],
                                [x - h / 2, y - r / 2],
                                [x - h / 2, y + r / 2],
                                [x, y + r],
                            ]
                        ],
                    }
                )
                polygons.append(hexagon)
                k=0
                    

    poly_grid = geopandas.GeoDataFrame({'geometry': polygons})
    poly_grid.plot(ax=base, facecolor="none", edgecolor='black', lw=0.7, zorder=15)
    poly_grid.to_file(os.getcwd()+'\\shapefiles\\grid_exa.shp')
