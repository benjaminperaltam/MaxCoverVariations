import numpy as np
from time import time
from multiprocessing.pool import Pool
from GAMixModel_var2 import capacity, timeConstr, reachablemask, J
path = 'Inputs/inds200_feasible.npy'

values = []
for v in reachablemask.values():
    values += list(v)

def init3d(N, NCS=165):

    _values, counts = np.unique(np.array(values), return_counts=True)
    _values[np.argpartition(counts, -2 * NCS)[-2 * NCS:]]
    CSs = np.random.choice(_values, size=NCS)
    out = np.zeros((J, 3))
    out[:, 0] = np.arange(J)
    out[:, 1] = np.nan
    out[:, 2] = np.nan
    time1, time2 = timeConstr(out)
    kcap = capacity(out)

    for j in range(J):
        timeavaiable = np.where((time2 < 0))[0] if (time2 < 0).max() \
            else np.where((time1 > 0))[0]
        CSaviable = np.where(kcap[:, timeavaiable].max(axis=1) > 0)[0]
        inter = np.intersect1d(reachablemask[j], CSs)
        inter = np.intersect1d(inter, CSaviable)
        if len(inter) > 0:
            i = np.random.choice(inter)
            t = np.random.choice(timeavaiable)

        else:
            i = np.nan
            t = np.nan

        out[j, 1] = i
        out[j, 2] = t

        time1, time2 = timeConstr(out)
        kcap = capacity(out)

    print("Individual created")
    
    return out

if __name__ == '__main__':
    pop = Pool(processes=16)
    init_time = time()
    out = pop.map(init3d, range(200))
    op_time = time() - init_time

    print("""
    End of (succesful) operation, {} individuals generated in {:.2f}[s]
    """.format(len(out), op_time))

    with open(path, 'wb') as f:
        np.save(f, np.array(out))

    print(f"Population saved in {path}")