import numpy as np
from santiago_map import santiago_map
import geopandas as gpd
import pandas as pd

N_cs = 1500 # Number of posible locations
xmin, ymin, xmax, ymax =  santiago_map.total_bounds

hx = (xmax - xmin) / int(N_cs ** 0.5)
hy = (xmax - xmin) / int(N_cs ** 0.5)

xbins = np.arange(xmin, xmax, hx)
ybins = np.arange(ymin, ymax, hy)

xy = np.array(np.meshgrid(xbins, ybins)).T.reshape(-1, 2)

points = gpd.GeoSeries.from_xy(xy[:, 0], xy[:, 1])

inmask = santiago_map.unary_union.contains(points)

while inmask.sum() < N_cs:
    hx *= 0.9
    hy *= 0.9
    xbins = np.arange(xmin, xmax, hx)
    ybins = np.arange(ymin, ymax, hy)
    xy = np.array(np.meshgrid(xbins, ybins)).T.reshape(-1, 2)
    points = gpd.GeoSeries.from_xy(xy[:, 0], xy[:, 1])
    inmask = santiago_map.unary_union.contains(points)
    print(f'{inmask.sum()} in the map with separation {int(hx)}m')

pd.DataFrame({
    'x': points[inmask].x,
    'y': points[inmask].y,
}).to_csv('Inputs/locations.csv')

