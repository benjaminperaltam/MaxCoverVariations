save=False

#if save:
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.integrate import dblquad

from demand import demand, edges, Origen, Destino, kde
from dists import manhattan

r = 5e2 # Integration radius [m]
N = 2527 # Sample size
recalculate_sample = False

if recalculate_sample:
    idxs = np.random.randint(0, int(len(demand)/2), N)
    np.save(f'Index_samples/N{N}.npy', idxs)
else:
    idxs = np.load(f'Index_samples/N{N}.npy', allow_pickle=True)

edges_sample = edges.iloc[idxs].reset_index()

#comunes_origin = edges_sample.Com_origin.unique()
#comunes_dest = edges_sample.Com_dest.unique()
#comunes = np.unique(np.concatenate((comunes_origin, comunes_dest)))
#mean_vel = pd.DataFrame(columns=comunes, index=comunes)
#for origin in comunes:
#    for dest in comunes:
#        mean_vel.loc[origin][dest] = Viajes_auto[
#            (Viajes_auto['ComunaOrigen'] == origin) & (Viajes_auto['ComunaDestino'] == dest) | \
#            (Viajes_auto['ComunaOrigen'] == dest) & (Viajes_auto['ComunaDestino'] == origin)
#        ]['dx/dt'].mean()

nodes = pd.concat((
    edges_sample[['Origin', 'Com_origin']].rename(columns={'Origin': 'Node', 'Com_origin': 'Comune'}),
    edges_sample[['Dest', 'Com_dest']].rename(columns={'Dest': 'Node', 'Com_dest': 'Comune'})
)).reset_index().drop('index', axis=1)
nodes = gpd.GeoDataFrame(nodes, geometry='Node', crs='epsg:32718')

OD = np.concatenate((edges_sample.Origin, edges_sample.Dest))
Com = np.concatenate((edges_sample.Com_origin, edges_sample.Com_dest))
edges_sample = pd.DataFrame({
    'Origin': OD.repeat(len(OD)),
    'Dest': np.tile(OD, len(OD)),
    'Com_origin': Com.repeat(len(OD)),
    'Com_dest': np.tile(Com, len(OD))
})


edges_sample['w'] = edges_sample.apply(lambda x: manhattan(x.Origin, x.Dest), axis=1)
edges_sample = edges_sample[edges_sample.w > 0].reset_index().drop('index', axis=1)

if recalculate_sample:
    nodes['f'] = nodes.apply(lambda n: dblquad(lambda y, x: kde.pdf((x, y))[0],
                                        n.Node.x - r, n.Node.x + r,
                                        lambda x: n.Node.y - np.sqrt(r**2 - (x - n.Node.x)**2),
                                        lambda x: n.Node.y + np.sqrt(r**2 - (x - n.Node.x)**2))[0],
                        axis=1)
    np.save(f'f_sample/N{N}.npy', nodes['f'].to_numpy())
else:
    f_ = np.load(f'f_sample/N{N}.npy')
    nodes['f'] = f_


#for idx1, origin in nodes.iterrows():
#    for idx2, dest in nodes.iterrows():
#        if len(edges_sample[(edges_sample['Origin'] == origin.Node) & (edges_sample['Dest'] == dest.Node)]) == 0:
#            distance = manhattan(origin.Node, dest.Node) * 1e-3
            #try:
            #    dx_dt = mean_vel.loc[origin.Comune][dest.Comune]
            #    w = distance / dx_dt
            #except KeyError:
            #    print((origin.Comune, dest.Comune))
            #    dx_dt = None
            #    w = None
                
#            new_row = {'Origin': [origin.Node],
#                       'Dest': [dest.Node],
#                       'Com_origin': [origin.Comune],
#                       'Com_dest': [dest.Comune],
#                       'w': [distance]
            #          'w': [w],
            #          'dx': [distance],
            #          'dx/dt': [dx_dt]
#                       }
#            new_row = pd.DataFrame(new_row)
#            edges_sample = pd.concat((edges_sample, new_row))

#edges_sample = edges_sample.reset_index().drop(['index'], axis=1)
#edges_sample = edges_sample[(edges_sample['w'] > 1e-4) & (~pd.isna(edges_sample['w']))]

#edges_sample['w'] = edges_sample.apply(lambda x: manhattan(x.Origin, x.Dest), axis=1)

D = pd.DataFrame({'O': Origen.iloc[idxs].reset_index().geometry,
                'D': Destino.iloc[idxs].reset_index().geometry,
                'f': nodes.iloc[:int(len(nodes)/2)].f,
                'r': 1e4,
                'tol': 0.2})
    
#    nodes.to_csv(f'Inputs/nodes_N{len(nodes)}.csv')
#    D.to_csv(f'Inputs/D_N{len(D)}.csv')

#else:
#        import numpy as np
#        import pandas as pd
#        from demand import edges
#        from shapely.geometry import Point
#        nodes = pd.read_csv('Inputs/nodes_N5054.csv')
#        D = pd.read_csv('Inputs/D_N2527.csv')
#        Tp = np.load(f'Inputs/Tp.npy', allow_pickle=True)
#        r = 5e2 # Integration radius [m]
#        N = 2527 # Sample size
#        recalculate_sample = False
#        D.O = D.O.apply(lambda x: Point([float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]))
#        D.D = D.D.apply(lambda x: Point([float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]))
    
#        if recalculate_sample:
#            idxs = np.random.randint(0, int(len(demand)/2), N)
#            np.save(f'Index_samples/N{N}.npy', idxs)
#        else:
#            idxs = np.load(f'Index_samples/N{N}.npy', allow_pickle=True)
        
#        edges_sample = edges.iloc[idxs].reset_index()