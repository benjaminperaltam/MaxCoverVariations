import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import gaussian_kde

from santiago_map import santiago_map
from Viajes_auto import df_viajes
from dists import manhattan


Origen = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df_viajes.x_origen,
                                                      df_viajes.y_origen),
                          crs='EPSG:32719')
Origen['Tipo'] = ['Origen'] * len(Origen)
#Origen['dt'] = df_viajes['dt'].reset_index().drop('index', axis=1)
Origen['Comuna'] = df_viajes['Comuna_origen'].reset_index().drop('index', axis=1).astype(str)

Destino = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df_viajes.x_destino,
                                                       df_viajes.y_destino),
                           crs='EPSG:32719')
Destino['Tipo'] = ['Destino'] * len(Destino)
#Destino['dt'] = df_viajes['dt'].reset_index().drop('index', axis=1)
Destino['Comuna'] = df_viajes['Comuna_destino'].reset_index().drop('index', axis=1).astype(str)

Origen.to_crs("EPSG:32718", inplace=True)
Destino.to_crs("EPSG:32718", inplace=True)

inmask_origen = santiago_map.unary_union.contains(Origen.geometry)
inmask_dest = santiago_map.unary_union.contains(Destino.geometry)
inmask = inmask_origen & inmask_dest

Origen = Origen[inmask]
Destino = Destino[inmask]

demand = pd.concat((Origen[inmask].reset_index().drop('index', axis=1),
                    Destino[inmask].reset_index().drop('index', axis=1))).reset_index().drop('index', axis=1)

edges = pd.DataFrame({'Origin': Origen.geometry,
                      'Dest': Destino.geometry,
                      'Com_origin': Origen.Comuna,
                      'Com_dest': Destino.Comuna
                      #'w': Origen.dt})
                      })
#edges['dx'] = np.array([np.abs(o.x - d.x) + np.abs(o.y - d.y) for (o, d) in zip(edges['Origin'], edges['Dest'])]) / 1000
#edges['w'] = edges.apply(lambda x: x['w'] + 0.8 if x['w'] == 0 else x['w'], axis=1)
edges['w'] = edges.apply(lambda x: manhattan(x.Origin, x.Dest), axis=1)
#edges['dx/dt'] = edges['dx'] / edges['w'] + 1e-3
edges = edges.reset_index().drop('index', axis=1)

X = np.array([[p.x, p.y] for p in demand.geometry])
kde = gaussian_kde(X.T)
