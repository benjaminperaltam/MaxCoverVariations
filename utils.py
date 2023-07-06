import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
from dists import dist, manhattan
from sample_demand import nodes, edges_sample

def reachable_area(p):
    d = dist(p.D, p.O)
    dx = -(p.D.y - p.O.y) / d
    dy = (p.D.x - p.O.x) / d
    dw = manhattan(p.O, p.D) * p.tol / 2

    X_point_list = [p.O.x - ((p.D.x - p.O.x) / d * dw),
                    p.O.x + dx * dw,
                    p.D.x + dx * dw,
                    p.D.x + ((p.D.x - p.O.x) / d * dw),
                    p.D.x - dx * dw,
                    p.O.x - dx * dw]
    
    Y_point_list = [p.O.y - ((p.D.y - p.O.y) / d * dw),
                    p.O.y + dy * dw,
                    p.D.y + dy * dw,
                    p.D.y + ((p.D.y - p.O.y) / d * dw),
                    p.D.y - dy * dw,
                    p.O.y - dy * dw]

    polygon_geom = Polygon(zip(X_point_list, Y_point_list))
    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:32718',
                               geometry=[polygon_geom])

    return polygon

def reachable_nodes(area, nodes=nodes):
    nodes_serie = gpd.GeoSeries(nodes['Node'])
    contained_mask = area.iloc[0].geometry.contains(nodes_serie)
    return nodes[contained_mask]

def feasible_arcs(o, area):
    #Arcs with origin in o and a reachable destination
    close_nodes = reachable_nodes(area).Node
    edges_ = edges_sample[(edges_sample['Origin'] == o) \
                          & edges_sample['Dest'].isin(close_nodes)]
    return pd.DataFrame({'O': edges_.Origin,
                        'D': edges_.Dest,
                        'w': edges_.w})

def feasible_paths(p, max_w, max_len=3, verb=0):
    max_w = (1 + p.tol) * manhattan(p.O, p.D)
    sols = pd.DataFrame(columns=['path', 'total_w'])

    paths = pd.DataFrame({'path': [p.O],
                        'last_node': [p.O],
                        'total_w': [0]})

    area = reachable_area(p)
    nodes_ = reachable_nodes(area)
    feasible_arcs_ = edges_sample[edges_sample.Origin.isin(nodes_.Node) \
                                  & (edges_sample.Dest.isin(nodes_.Node))]
    feasible_arcs_ = feasible_arcs_[feasible_arcs_.w <= max_w]

    paths = paths.merge(feasible_arcs_[feasible_arcs_.Origin == p.O],
                        left_on='last_node',
                        right_on='Origin',
                        how='left')

    paths['last_node'] = paths['Dest']
    paths['total_w'] = paths.apply(lambda x: x['total_w'] + x['w'], axis=1)
    paths['path'] = paths[['path', 'Dest']].values.tolist()
    paths = paths[['path', 'last_node', 'total_w']]

    niter = 0
    while len(paths) > 0 and niter < max_len:
        paths = paths[paths['total_w'] <= max_w]
        paths = paths[~paths.apply(lambda x: x.last_node in x.path[:-1], axis=1)]
        sols = pd.concat((
            sols,
            paths[paths['last_node'] == p.D][['path', 'total_w']]
        ))
        paths = paths[paths['last_node'] != p.D]
        if verb > 0:
            print(f'Iter {niter}, nsols: {len(sols)}, unfinished_paths: {len(paths)}')
        
        last_nodes = paths['last_node'].unique()

        paths = paths.merge(
            feasible_arcs_[feasible_arcs_.Origin.isin(last_nodes)],
            left_on='last_node',
            right_on='Origin',
            how='left'
        )

        paths['last_node'] = paths['Dest']
        paths['total_w'] = paths.apply(lambda x: x['total_w'] + x['w'], axis=1)
        paths.path = paths.apply(lambda x: x['path'] + [x['Dest']], axis=1)
        paths = paths[['path', 'last_node', 'total_w']]
        niter += 1

    return sols