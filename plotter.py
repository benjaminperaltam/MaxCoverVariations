"""
Class for plotting assignation results (from genetic algorithms solutions)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plotter(object):
    def __init__(self, assignation, figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        self.z = assignation
        self.ax = ax
        self.fig = fig

    def z_from_dummy(self):
        z_dummy = self.z
        J = z_dummy.shape[1]
        z_oneencode = np.zeros((J, 3))
        z_oneencode[:, 0] = np.arange(J)
        z_oneencode[:, 1] = np.nan
        z_oneencode[:, 2] = np.nan

        active_idxs = np.where(z_dummy == 1)

        for (i, j, t) in active_idxs:
            z_oneencode[j, 1] = i
            z_oneencode[j, 2] = t
        
        self.z = z_oneencode
        
        return z_oneencode

    def base_map(self):
        from santiago_map import santiago_map

        ax = self.ax
        santiago_map.boundary.plot(ax=ax, color='black', linewidth=.5)

    def demand_nodes(self):
        from sample_demand import nodes

        ax = self.ax
        nodes.plot(edgecolor='gray',
                   facecolor='none',
                   alpha=.2,
                   ax=ax,
                   label=f'Demand nodes ({len(nodes)})')
        
    def active_CS(self):
        from sample_demand import nodes
        import numpy as np

        ax = self.ax
        z = self.z
        activeIndex = np.unique(z[:, 1][~np.isnan(z).max(axis=1)])
        activeIndex = activeIndex.astype(int)
        nodes[activeIndex].plot(facecolor='darkred',
                                edgecolor='black',
                                marker='D',
                                ax=ax,
                                label=f'Active CS ({len(activeIndex)})')
        
    def active_CS_wnumber(self):
        from sample_demand import nodes

        ax = self.ax
        z = self.z
        activeIndex = np.unique(z[:, 1][~np.isnan(z).max(axis=1)])
        activeIndex = activeIndex.astype(int)
        nodes[activeIndex].plot(cmap='Reds',
                                edgecolor='black',
                                marker='D',
                                ax=ax,
                                label=f'Active CS ({len(activeIndex)})',
                                c=[(z.astype(int)[:, 1] == i).sum() \
                                   for i in activeIndex])
    
    def satisfied_demand(self):
        import pickle
        from graphs import Demand, DemandNode

        ax = self.ax
        z = self.z

        D = pd.read_csv('Inputs/D_N2527.csv')
        with open('Inputs/reachable_nodes.pkl', 'rb') as fp:
            reachablemask = pickle.load(fp)
        
        demand_ = Demand(D.apply(lambda x: DemandNode(O=x.O,
                                                      D=x.D,
                                                      f=x.f,
                                                      tol=x.tol),
                                                      axis=1).to_numpy())
        J = len(demand_)
        assignationTime = ~np.isnan(z).max(axis=1)
        reachable_ = np.array([z[j, 1] in reachablemask[j] \
                               for j in range(J)])
        _y = (assignationTime & reachable_)

        ax.scatter([d.O.x for d in demand_[_y]],
                   [d.O.y for d in demand_[_y]],
                   color='green',
                   label=f'Covered demand ({2 * sum(_y)})')
        
        ax.scatter([d.D.x for d in demand_[_y]],
                   [d.D.y for d in demand_[_y]],
                   color='green')
    
    def plot_assignation(self, path=''):
        ax = self.ax
        z = self.z

        covered_nodes = (~np.isnan(z).max(axis=1)).sum()
        ax.set_title(f'Covered OD pairs: {covered_nodes}/{z.shape[0]}')

        self.base_map()
        self.demand_nodes()
        self.satisfied_demand()
        self.active_CS()
        ax.legend()

        fig = self.fig
        fig.show()

        if path != '':
            fig.savefig(path)

    def time_distribution(self, T=12):
        from datetime import datetime, timedelta
        import matplotlib.dates as mdates

        z = self.z
        ax = self.ax
        fig = self.fig
        base_hour = datetime(2023, 1, 1, 0, 0, 0)
        arr = np.array([base_hour + int(24/T) * timedelta(hours=t) for t in range(T)]).astype('datetime64')
        ax.bar(x=np.sort(arr),
            height=np.unique(z[:, 2], return_counts=True)[1][:12],
            width=120./24/60,
            edgecolor='black',
            align='edge')
        ax.set_title(f'Number of vehicles per hour')
        hours = mdates.HourLocator(interval = 2)
        h_fmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)
        plt.xticks(rotation = 45)
        fig.show()