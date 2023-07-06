import numpy as np
import pandas as pd
from sample_demand import nodes
from shapely import is_empty
from shapely.geometry import Point

n_long, n_lat = 25, 25

xbins = np.linspace(min(nodes.Node.x), max(nodes.Node.x), n_long)
ybins = np.linspace(min(nodes.Node.y), max(nodes.Node.y), n_lat)

def get_square(coordx, coordy, xbins=xbins, ybins=ybins):
    hx = xbins[1] - xbins[0]
    hy = ybins[1] - ybins[0]
    return int((coordx - xbins[0])/hx), int((coordy - ybins[0])/hy)

nodes['Disc'] = nodes.Node.apply(lambda n: get_square(n.x, n.y))

D_disc = pd.DataFrame(columns=['O', 'D', 'f_origin', 'f_dest',
                               'r', 'tol', 'N'])

for i, j in nodes.Disc.unique():
    origen_ij = nodes[:int(len(nodes)/2)][nodes[:int(len(nodes)/2)]['Disc'] == (i, j)]
    destino_ij = nodes.iloc[(origen_ij.index + int(len(nodes)/2))]
    D_disc = pd.concat((
        D_disc,
        pd.DataFrame({
            'O': Point(origen_ij.Node.x.mean(), origen_ij.Node.y.mean()),
            'D': Point(destino_ij.Node.x.mean(), destino_ij.Node.y.mean()),
            'f_origin': origen_ij.f.sum(),
            'f_dest': destino_ij.f.sum(),
            'r': 1e4,
            'tol': 0.2,
            'N': len(origen_ij)
        }, index=[(i, j)])
    ))

D_disc = D_disc[~(D_disc.O.apply(is_empty) | D_disc.D.apply(is_empty))]