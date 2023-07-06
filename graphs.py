
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import copy
from shapely.geometry import Polygon
from tqdm import tqdm

#from sample_demand import nodes
from shapely.geometry import Point
import pandas as pd
nodes = pd.read_csv('Inputs/nodes_N5054.csv')
D = pd.read_csv('Inputs/D_N2527.csv')
Tp = np.load(f'Inputs/Tp.npy', allow_pickle=True)
D.O = D.O.apply(lambda x: Point([float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]))
D.D = D.D.apply(lambda x: Point([float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]))
nodes.Node = nodes.Node.apply(lambda x: Point([float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]))
nodes = gpd.GeoDataFrame(nodes, crs="EPSG:32718", geometry='Node')

from dists import manhattan, dist


class Path(object):

    def __init__(self, nodes=np.array([])) -> None:
        self.nodes = nodes

    def append(self, node):
        self.nodes = np.concatenate((self.nodes, [node]))

    def plot(self, fig=None, od_marker=None, label=None, **kwards):
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax = fig.get_axes()[0]
        nodes = self.nodes
        xs = np.vectorize(lambda n: n.x)(nodes)
        ys = np.vectorize(lambda n: n.y)(nodes)
        ax.plot(xs, ys, label=label, **kwards)
        ax.scatter([nodes[0].x, nodes[-1].x],
                   [nodes[0].y, nodes[-1].y],
                   marker=od_marker,
                   color='black')

    def contains(self, node):
        return node in self.nodes
    
    def w(self, dist_function=manhattan):
        total_len = 0
        nodes = self.nodes
        for k in range(1, len(nodes)):
            total_len += dist_function(nodes[k-1], nodes[k])
        return total_len
    
    def w_deviation(self, dist_function=manhattan):
        return (self.w(dist_function) 
                - dist_function(self.nodes[0], self.nodes[-1]))\
                      / self.w(dist_function)
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, key):
        return self.nodes[key]

    def has_loop(self):
        return (len(self) > 1) and (self.nodes[-1] in self.nodes[:-1])
    
    def __str__(self):
        return str(self.nodes)
    
    def remove(self, key):
        self.nodes = np.delete(self.nodes, key)
    
    def copy(self):
        return copy.deepcopy(self)
    
    
class PathCollection(object):

    def __init__(self, paths=np.array([], dtype=object)) -> None:
        self.paths = paths

    def contains(self, n):
        return np.array([p.contains(n) for p in self.paths])
    
    def from_series(self, paths_serie):
        self.paths = np.array([Path(p) for p in paths_serie], dtype=object)
    
    def __getitem__(self, key):
        return self.paths[key]
    
    def __len__(self):
        return len(self.paths)
    
    def w(self, dist_function=manhattan):
        return np.vectorize(lambda p: p.w(dist_function))(self.paths)
    
    def append(self, path):
        out = np.empty(len(self) + 1, dtype=object)
        out[:-1] = self.paths
        out[-1] = path
        self.paths = out

    def has_loop(self):
        return np.vectorize(lambda p: p.has_loop())(self.paths)
    
    def remove(self, key):
        self.paths = np.delete(self.paths, key)

    def plot(self, fig=None, label=None):
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax = fig.get_axes()[0]

        if max(self.w()) > min(self.w()):
            color_values = (self.w() - min(self.w()))
            pp_color_values = color_values / max(self.w())
            colors = plt.cm.Reds(color_values)
        else:
            colors = len(self.paths) * ['red']
            pp_color_values = [0]

        for k, path in enumerate(self.paths):
            path.plot(fig=fig,
                      color=colors[k],
                      label=(label if k == 0 else None))

        sm = plt.cm.ScalarMappable(cmap='Reds',
                                   norm=plt.Normalize(vmin=min(pp_color_values * 100),
                                                      vmax=max(pp_color_values * 100)))
        cbar = fig.colorbar(sm, ax=ax, fraction=.047, pad=0.04)
        cbar.set_label(r'Extra travel (%)')
    
    def __str__(self):
        report = ''
        for p in self.paths:
            report += ('\n' + p.__str__())
        return report
    

class DemandNode(object):

    def __init__(
            self, O, D, tol, f, paths=PathCollection(paths=np.array([]))
    ) -> None:
        self.O = O
        self.D = D
        self.f = f
        self.tol = tol
        self.paths = paths

    def w(self, dist_function=manhattan):
        return dist_function(self.O, self.D)
    
    def __str__(self):
        out_str = f"""Origin: {self.O} \nDestination: {self.D} \nFlow: {self.f}\nTolerance: {self.tol}
        """
        return out_str
    
    def get_paths(self, max_len=3, verb=0):
        max_w = (1 + self.tol) * self.w()
        self.paths = PathCollection()
        nodes_ = self.reachable_nodes().reset_index().Node
        nodes_ = nodes_[nodes_ != self.O]
        #sols = PathCollection(paths = np.array([Path(nodes=np.array([self.O, self.D]))]))
        sols = PathCollection()

        #for n in nodes_[nodes_ != self.D]:
        #    self.paths.append(Path(nodes=np.array([self.O, n])))
        self.paths.append(Path(nodes=np.array([self.O])))
        
        niter = 0
        while len(self.paths) > 0 and niter <= max_len:
            npaths = len(self.paths)
            remove_idxs = []
            for kp, p in enumerate(self.paths.paths):
                if (p.w() > max_w) or (p.has_loop()):
                    remove_idxs.append(kp)
                elif (p.w() <= max_w) and p[-1] == self.D:
                    sols.append(p.copy())
                    remove_idxs.append(kp)
                else:
                    if niter <= max_len - 1:
                        for n in nodes_:
                            new_path = p.copy()
                            new_path.append(n)
                            self.paths.append(new_path)
                        remove_idxs.append(kp)
                    
            self.paths.remove(remove_idxs)

            if verb > 0:
                print(f'Iter {niter}, nsols: {len(sols)}, unfinished_paths: {npaths}')

            niter += 1
        
        self.paths = sols
        return sols
  
    def reachable_area(self):
        p = self
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
        polygon = gpd.GeoDataFrame(index=[0],
                                   crs='epsg:32718',
                                   geometry=[polygon_geom])

        return polygon

    def reachable_nodes(self, nodes=nodes):
        area = self.reachable_area()
        contained_mask = area.iloc[0].geometry.contains(gpd.GeoSeries(nodes['Node']))
        return nodes[contained_mask]

    def plot_paths(self, sample_size=-1, fig=None, label='Feasible paths'):
        sample_idxs = np.arange(0, len(self.paths))
        if sample_size >= 0:
            sample_idxs = np.random.choice(a=sample_idxs, size=sample_size, replace=False)

        label += f' ({len(sample_idxs)})'
        PathCollection(self.paths[sample_idxs]).plot(fig=fig, label=label)


class Demand(object):

    def __init__(self, nodes=np.array([])) -> None:
        self.nodes = nodes
    
    def append(self, node):
        out = np.empty(len(self) + 1, dtype=object)
        out[:-1] = self.nodes
        out[-1] = node
        self.nodes = out

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        return self.nodes[key]

    def remove(self, key):
        self.ndoes = np.delete(self.nodes, key)

    def __str__(self):
        out = ''
        for n in self.nodes:
            out += '\n' + n.__str__()
        return out
    
    def get_paths(self, max_len=3):
        for n in tqdm(self.nodes):
            n.get_paths(max_len=max_len)
