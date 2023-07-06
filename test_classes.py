import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphs import Path, PathCollection, Demand, DemandNode
from shapely.geometry import Point

from sample_demand import D

def test_Path():
    print('Test Path class:')
    path_ = Path(D.O[:5].to_numpy())
    print(f'\nPath: {path_}')
    path_.append(D.O[6])
    print(f'\nPath with extra node: {path_}')
    print(f'\n Contains its first node: {path_.contains(path_[0])}')
    print(f'\n Does not contains an extra node: {path_.contains(D.O[7])}')
    print(f'\nLength of the path (number of nodes): {len(path_)}')
    print(f'\nLength of the path (Manhattan dist.): {path_.w()}')
    print(f'\nExtra trave w/r to direct trip: {path_.w_deviation()}')
    print(f'\nThe path has no loops: {path_.has_loop()}')
    path_.remove(0)
    print(f'\nPath w/o the first node: {path_}')
    copy_ = path_.copy()
    copy_.append(Point(1,1))
    print(f'\n Original path: {path_}')
    print(f'\nCopy: {copy_}')
    print('\nPlot:')
    fig, ax = plt.subplots(figsize=(10, 10))
    path_.plot(fig=fig, label='Example label')
    ax.legend()
    fig.show()

def test_PathCollection():
    print('Test PathCollection class:')
    path1 = Path(D.O[:3].to_numpy())
    path2 = Path(D.O[3:8].to_numpy())
    path3 = Path(D.O[8:9].to_numpy())
    path4 = Path(D.O[9:16].to_numpy())
    collection_ = PathCollection(
        np.array([path1, path2, path3], dtype=object)
    )
    print(f'\nCollection: {collection_}')
    collection_.from_series(pd.Series([D.O[:3].to_numpy(),
                                       D.O[3:8].to_numpy(),
                                       D.O[8:9].to_numpy()]))
    print(f'\nCollection from serie: {collection_}')
    collection_.append(path4)
    print(f'\nCollection with new path: {collection_}')
    print(f'\nAny element have loops: {collection_.has_loop()}')

    collection_.remove(0)
    print(f'\nCollection w/o its first element: {collection_}')
    print(f'\nPaths lengths (Manhattan dist.):{collection_.w()}')
    print('\nPlot:')
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    collection_.plot(fig=fig2, label='Example label')
    ax2.legend()
    fig2.show()

def test_DemandNode():
    print('Test DemandNode class:')
    demandnode_ = DemandNode(O=D.iloc[0].O,
                             D=D.iloc[0].D,
                             f=D.iloc[0].f,
                             tol=D.iloc[0].tol)
    print(demandnode_)
    print(f'\nDistance: {demandnode_.w()}')
    demandnode_.get_paths(verb=1)

def test_Demand():
    print('Test Demand class:')
    demand_ = Demand(D.apply(lambda x: DemandNode(O=x.O,
                                                  D=x.D,
                                                  f=x.f,
                                                  tol=x.tol),
                                                  axis=1).to_numpy())
    print(f'\nLength demand {len(demand_)}')
    print(f'\nNodes: {demand_}')

def main():
    test_Path()
    test_PathCollection()
    test_DemandNode()
    test_Demand()

if __name__ == '__main__':
    main()