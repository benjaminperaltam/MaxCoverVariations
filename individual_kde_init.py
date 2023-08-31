import numpy as np
from time import time
import os
from multiprocessing.pool import Pool
import pickle
from stochastical_models.GAMixStochModel import capacity, timeConstr
path = 'stochastical_models/inds200_feasible_kde.npy'

J = 2527
N = 10
nthreads = 12

with open('stochastical_models/reachable_nodes.pkl', 'rb') as fp:
    reachablemask = pickle.load(fp)

values = []
for v in reachablemask.values():
    values += list(v)

def init3d(_, NCS=165):

    _values, counts = np.unique(np.array(values), return_counts=True)
    _values[np.argpartition(counts, -2 * NCS)[-2 * NCS:]]
    CSs = np.random.choice(_values, size=NCS)
    out = np.zeros((J, 3, N))

    out = np.zeros((J, 3, N))
    out[:, 0, :] = np.arange(J).repeat(N).reshape(J, N)
    out[:, 1:, :] = np.nan
    time1, time2 = timeConstr(out)
    kcap = capacity(out)

    for n in range(N):
        for j in range(J):
            timeavaiable = np.where((time2[:, n] < 0))[0] if (time2[:, n] < 0).max() \
                else np.where((time1[:, n] > 0))[0]
            CSaviable = np.where(kcap[:, timeavaiable, n].max(axis=1) > 0)[0]
            inter = np.intersect1d(reachablemask[(j, n)], CSs)
            inter = np.intersect1d(inter, CSaviable)

            if len(inter) > 0:
                i = np.random.choice(inter)
                t = np.random.choice(timeavaiable)
                time1, time2 = timeConstr(out)
                kcap = capacity(out)

            else:
                i = np.nan
                t = np.nan

            out[j, 1, n] = i
            out[j, 2, n] = t

    print("Individual created")
    
    return out

if __name__ == '__main__':
    pop = Pool(processes=nthreads)

    if os.path.isfile(path):
        out0 = np.load(path)
        current_len = out0.shape[0]
    else:
        current_len = 0
    
    for k in range((200 - current_len) // nthreads
                   + int((200 - current_len) % nthreads > 0)):
        
        try:
            init_time = time()
            out = pop.map(init3d, range(nthreads))
            op_time = time() - init_time

            print("""
            End of (succesful) operation, {} individuals generated in {:.2f}[s]
            """.format(len(out), op_time))

            if os.path.isfile(path):
                out0 = np.load(path)
                out = np.concatenate((out0, out), axis=0)
            
            with open(path, 'wb') as f:
                np.save(f, np.array(out))

            print(f"Population saved in {path}")
        
        except KeyboardInterrupt:
            break