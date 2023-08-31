import numpy as np
from time import time
import pickle
from multiprocessing.pool import Pool
from GAMixModel_week import y_day, timeConstr, capacity, I, J, time_slots_demand
from tqdm.contrib.concurrent import process_map

path = 'week_model/inds200_feasible_w_slots.npy'

n_threads = 12
n_inds = 200

with open('week_model/reachable_nodes.pkl', 'rb') as f:
    reachablemask = pickle.load(f)

with open(f'Inputs/energy_demand_J{J}.pkl','rb') as f:
    energy_demand = pickle.load(f)

with open(f'Inputs/CS_potency_I{I}.pkl','rb') as f:
    CS_potency = pickle.load(f)

hours_demanded = (energy_demand / CS_potency)
time_slots = hours_demanded.sum(axis=-1) // .5 + 1
time_slots[hours_demanded.sum(axis=-1) == 0] = 0

values = []
for v in reachablemask.values():
    values += list(v)

def heuristic_solution(n, NCS=165):
    """
    Version w/o energy demand and time slots
    """

    # Chose the 2 * NCS most reachable charging stations 
    _values, counts = np.unique(np.array(values).reshape((1, -1)),
                                return_counts=True)
    _values[np.argpartition(counts, -2 * NCS)[-2 * NCS:]]
    
    # Chose NCS CS to activate
    CSs = np.random.choice(_values, size=NCS)
    out = np.zeros((J, 3, 7))
    out[:, 0, :] = np.arange(J).repeat(7).reshape(J, 7)
    out[:, 1:, :] = np.nan
    time1, time2 = timeConstr(out)
    kcap = capacity(out)

    s0 = np.random.randint(0, 7)

    for j in range(J):
        timeavaiable = np.where((time2[:, s0] < 0))[0] if (time2[:, s0] < 0).max() \
            else np.where((time1[:, s0] > 0))[0]
        CSaviable = np.where(kcap[:, timeavaiable, s0].max(axis=1) > 0)[0]
        inter = np.intersect1d(reachablemask[(j, s0)], CSs)
        inter = np.intersect1d(inter, CSaviable)

        if len(inter) > 0:
            i = np.random.choice(inter)
            t = np.random.choice(timeavaiable)

        else:
            i = np.nan
            t = np.nan

        out[j, 1, s0] = i
        out[j, 2, s0] = t

        time1, time2 = timeConstr(out)
        kcap = capacity(out)

    y = y_day(out).max(axis=-1)
    unexplored_s = np.delete(np.arange(7), s0)
    while min(y) == 0 and len(unexplored_s) > 0:
        si = np.random.choice(unexplored_s)
        for j in np.where(y == 0)[0]:
            timeavaiable = np.where((time2[:, si] < 0))[0] if (time2[:, si] < 0).max() \
                else np.where((time1[:, si] > 0))[0]
            CSaviable = np.where(kcap[:, timeavaiable, si].max(axis=1) > 0)[0]
            inter = np.intersect1d(reachablemask[(j, si)], CSs)
            inter = np.intersect1d(inter, CSaviable)

            if len(inter) > 0:
                i = np.random.choice(inter)
                t = np.random.choice(timeavaiable)

            else:
                i = np.nan
                t = np.nan

            out[j, 1, si] = i
            out[j, 2, si] = t

            time1, time2 = timeConstr(out)
            kcap = capacity(out)

        y = y_day(out).max(axis=-1)
        unexplored_s = np.delete(unexplored_s, np.where(unexplored_s == si)[0])

    print("Individual created")
    
    return out

def heuristic_solution(n, NCS=165):
    """
    Version w/ energy demand and time slots
    """
    # Chose the 2 * NCS most reachable charging stations 
    _values, counts = np.unique(np.array(values).reshape((1, -1)),
                                return_counts=True)
    _values[np.argpartition(counts, -2 * NCS)[-2 * NCS:]]
    
    # Chose NCS CS to activate
    CSs = np.random.choice(_values, size=NCS)

    # Array for output
    out = np.zeros((J, 3, 7))
    out[:, 0, :] = np.arange(J).repeat(7).reshape(J, 7)
    out[:, 1:, :] = np.nan

    # Time constrints
    time1, time2 = timeConstr(out)
    
    # Capacity constrint
    kcap = capacity(out)

    for j in range(J):
        unexplored_s = np.arange(0, 7, 1)

        # Iterate until every day is explored or the energy demand is covered
        while len(unexplored_s) > 0 and time_slots_demand(out)[j] < 0:
            
            # Chose a random day
            si = np.random.choice(unexplored_s)

            # Avaiable time slots
            timeavaiable = np.where((time2[:, si] < 0))[0] if (time2[:, si] < 0).max() \
                else np.where((time1[:, si] > 0))[0]
            
            # CSs with free slots
            CSaviable = np.where(kcap[:, timeavaiable, si].max(axis=1) > 0)[0]
            
            # Intersect reachable nodes, active CSs and CSs with free slots
            inter = np.intersect1d(reachablemask[(j, si)], CSs)
            inter = np.intersect1d(inter, CSaviable)

            # If the intersection is not empty, chose a random timeslot in an
            # avaiable CS and set an assignation
            if len(inter) > 0:
                i = np.random.choice(inter)
                t = np.random.choice(timeavaiable)

                time1, time2 = timeConstr(out)
                kcap = capacity(out)

            # Else, the vehicle is not assigned
            else:
                i = np.nan
                t = np.nan

            out[j, 1, si] = i
            out[j, 2, si] = t
            
            # Remove the day from the unexplored days list
            unexplored_s = np.delete(unexplored_s, np.where(unexplored_s == si)[0])

    print("Individual created")
    
    return out

if __name__ == '__main__':
    #pop = Pool(processes=n_threads)
    init_time = time()
    #out = pop.map(heuristic_solution, range(n_inds))
    out = process_map(heuristic_solution, range(n_inds), max_workers=n_threads)
    op_time = time() - init_time

    print("""
    End of (succesful) operation, {} individuals generated in {:.2f}[s]
    """.format(len(out), op_time))

    with open(path, 'wb') as f:
        np.save(f, np.array(out))

    print(f"Population saved in {path}")