"""
Excecute with command python -m scoop GAMixModel_week.py
Tho choose the number of workers python -m scoop -n 12 GAMixModel_week.py
must be in deap_env

individuals have the format:
ind = [[[1, 1, 2], [2, 5, 20], [3, None, None], ...], [], ...]
In this case, the vehicle 1 is assignated to the CS 1 in the time 2,
the vehicle 2 is assignated to the CS 5 in the time 20 and the vehicle 3
is not assignated to any CS.
"""

import random
import numpy as np
import pandas as pd
import pickle
import warnings
import json

warnings.filterwarnings("ignore")

from time import time
from numpy.linalg import norm
from functools import reduce

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scoop import futures

from graphs import Demand, DemandNode
from hist_time import Tp
from demand import demand
from shapely.geometry import Point
import geopandas as gpd

from dists import manhattan
from performance import Rendimiento_EV
from CSs import CSs

with open('params.json') as p:
    params = json.load(p)

J = params['CommonParams']['J']
Nce = params['CommonParams']['Nce']
n = params['GAMixModel_week']['n']
recalculate_sample = params['GAMixModel_week']['recalculate_sample']
calculate_reachable_nodes = params['GAMixModel_week']['calculate_reachable_nodes']
calculate_energy = params['GAMixModel_week']['calculate_energy']
calculate_potency = params['GAMixModel_week']['calculate_potency']
inds_path = params['GAMixModel_week']['inds_path']

Tp = Tp[::2] + Tp[1::2]
locations = pd.read_csv('Inputs/locations.csv')
locations = gpd.points_from_xy(locations.x, locations.y)
locations = gpd.GeoDataFrame(pd.DataFrame({'Node': locations}),
                             geometry='Node', crs="EPSG:32718")

D_total = pd.DataFrame({
    'O': demand[demand['Tipo'] == 'Origen'].reset_index().geometry,
    'D': demand[demand['Tipo'] == 'Destino'].reset_index().geometry,
    'r': [10000.0] * int(len(demand)/2),
    'tol': [0.2] * int(len(demand)/2) 
})

f_total = pd.read_csv('Inputs/f_total.csv')['0']
D_total['f'] = f_total

demand_total_ = Demand(D_total.apply(lambda x: DemandNode(O=x.O, D=x.D, f=x.f,
                                                          tol=x.tol),
                                                          axis=1).to_numpy())

if recalculate_sample:
    week_idxs = np.random.randint(0, int(len(demand)/2), size=(7, J))
    np.save('week_model/week_idxs.npy', week_idxs)
else:
    week_idxs = np.load('week_model/week_idxs.npy')

f_week = np.array([f_total[week_idxs[i]] for i in range(7)]).sum(axis=0)
sample_ = [demand_total_[week_idxs[i, :]] for i in range(7)]

for i in range(len(sample_)):
    NReachable = np.array([
        len(sample_[i][k].reachable_nodes(nodes=locations)) \
            for k in range(len(sample_[i]))
    ])

    for k in np.where(NReachable == 0)[0]:
        sample_[i][k].tol = 2 * sample_[i][k].tol

NReachable = np.array([
    [len(sample_[i][k].reachable_nodes(nodes=locations)) \
        for k in range(len(sample_[i]))] for i in range(len(sample_))
])

sample_ = [sample_[i][NReachable.max(axis=0) > 0] for i in range(len(sample_))]

I = len(locations)
J = sample_[0].shape[0]
T = len(Tp)
K = 3 * int(48 / T) * np.ones(I)

if calculate_energy:
    performance = np.random.choice(Rendimiento_EV['Autonomia'],
                                   size=len(sample_[0]), replace=True)
    
    dists = np.array([[manhattan(sample_[s][j].O, sample_[s][j].D)
                       for s in range(7)]
                       for j in range(len(sample_[0]))])
    
    energy_demand = dists / 1e3 / (performance.repeat(7).reshape(-1, 7))

    with open(f'Inputs/energy_demand_J{J}.pkl','wb') as f:
        pickle.dump(energy_demand, f)

else:
    with open(f'Inputs/energy_demand_J{J}.pkl','rb') as f:
        energy_demand = pickle.load(f)

if calculate_potency:
    CS_potency = CSs['Potencia (en kW) cargador'].values
    with open(f'Inputs/CS_potency_I{I}.pkl','wb') as f:
        pickle.dump(np.random.choice(CS_potency, size=(J, 7)), f)

else:
    with open(f'Inputs/CS_potency_I{I}.pkl','rb') as f:
        CS_potency = pickle.load(f)

hours_demanded = (energy_demand / CS_potency)
time_slots = hours_demanded.sum(axis=-1) // .5 + 1
time_slots[hours_demanded.sum(axis=-1) == 0] = 0


if calculate_reachable_nodes:
    reachablemask = {(j, s): sample_[s][j].reachable_nodes(nodes=locations).index \
                    for j in range(J) for s in range(7)}
    with open('week_model/reachable_nodes.pkl', 'wb') as fp:
        pickle.dump(reachablemask, fp)
        print('dictionary saved successfully to file')
else:
    with open('week_model/reachable_nodes.pkl', 'rb') as fp:
        reachablemask = pickle.load(fp)

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("map", futures.map)

def y(individual):
    return (time_slots_demand(individual) >= 0).astype(int)

def x(individual):
    out = np.zeros(I)
    active_index = np.unique(individual[:, 1, :][~np.isnan(individual[:, 1, :])]).astype(int)
    out[active_index] = 1
    return out

def nce(individual):
    return Nce - x(individual).sum()

def y_day(individual):
    assignationTime = ~np.isnan(individual).max(axis=1)
    reachable_ = np.array([[individual[j, 1, s] in reachablemask[(j, s)]
                            for s in range(7)] for j in range(J)])
    return (assignationTime & reachable_).astype(int)

def n_y(individual):
    return y_day(individual).sum(axis=-1)

def time_slots_demand(individual):
    return n_y(individual) - time_slots

def capacity(individual):
    """
    out dims (I, 12, 7)
    """
    CS = np.arange(I)
    out = np.zeros((I, T, 7))
    ind_ = individual.astype(int)
    for t in range(T):
        for s in range(7):
            tindividual = individual[(ind_[:, 2, s] == t)]
            count = np.array([(tindividual[:, 1, s] == i).sum() for i in CS])
            out[:, t, s] = K - count
    return out

def timeConstr(individual):
    """
    Out dim ((7, 12), (7, 12))
    """
    ind_ = individual.astype(int)
    out = np.zeros((T, 7))
    for t in range(T):
        for s in range(7):
            out[t, s] = (ind_[:, 2, s] == t).sum()
    return (np.ones((T, 7)) * J * Tp.repeat(7).reshape((T, 7)) * 1.1 - out,
            out - np.ones((T, 7)) * J * Tp.repeat(7).reshape((T, 7)) * .9)
    
def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    time1, time2 = timeConstr(individual)
    return (capacity(individual) >= 0).min() \
        & (nce(individual) >= 0) \
            & (time1 >= 0).min()

def distance(individual):
    time1, time2 = timeConstr(individual)
    return norm(capacity(individual)) + \
        norm(nce(individual)) + \
            norm(time1) + \
                norm(time2)

def distance(individual):
    """
    Distance to the feasible space
    """
    time1, time2 = timeConstr(individual)
    return norm(capacity(individual)) + \
        norm(nce(individual)) + \
            norm(time1)

def unsatisfied_demand(individual):
    z = individual
    satisfied_demand = y(z)
    return len(satisfied_demand) - satisfied_demand.sum()

def day_customMultiFlip(individual, indpb, s):
    CSMask = np.random.choice(a=[True, False], size=J, p=[indpb, 1 - indpb])
    TimeMask = np.random.choice(a=[True, False], size=J, p=[indpb, 1 - indpb])

    for j in range(J):
        if CSMask[j]:
            individual[j] = np.random.choice(np.concatenate((reachablemask[(j, s)],
                                                             np.array([np.nan]))))
    
    individual[TimeMask, 2] = np.random.choice(np.concatenate((np.arange(T),
                                                               np.array([np.nan]))),
                                               size=sum(TimeMask))
    
    individual[np.isnan(individual[:, 1]), 2] = np.nan
    individual[np.isnan(individual[:, 2]), 1] = np.nan

    return individual

def day_customTwoPoint(ind1, ind2):
    size = min(ind1.shape[0], ind2.shape[0])
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    piece1 = ind1[:, 1:][cxpoint1:cxpoint2]
    piece2 = ind2[:, 1:][cxpoint1:cxpoint2]

    ind1[:, 1:][cxpoint1:cxpoint2] = piece2
    ind2[:, 1:][cxpoint1:cxpoint2] = piece1

    return ind1, ind2

def customMultiFlip(individual, indpb):
    for s in range(7):
        individual[:, :, s] = day_customMultiFlip(individual[:, :, s],
                                                  indpb, s)
    return individual,

def customTwoPoint(ind1, ind2):
    for s in range(7):
        ind1[:, :, s], ind2[:, :, s] = day_customTwoPoint(ind1[:, :, s],
                                                          ind2[:, :, s])
    return ind1, ind2

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "rb") as pop_file:
        contents = np.load(pop_file)
    return pcls((ind_init(c)) for c in contents)

def objective_function(individual):
    return y(individual) @ f_week

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", objective_function)

# register the crossover operator
toolbox.register("mate", customTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.1
toolbox.register("mutate", customMultiFlip, indpb=0.01)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, -1, distance))

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess",
                 initPopulation,
                 list,
                 toolbox.individual_guess,
                 inds_path)


def main():
    try:
        random.seed(42)

        # create an initial population of individuals
        pop = toolbox.population_guess()
        out = tools.HallOfFame(1, similar=np.array_equal)
        out.update(pop)
        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            try:
                ind.fitness.values = (fit, )
            except TypeError:
                ind.fitness.values = (fit[0], )

        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of 
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        out_df = pd.DataFrame(columns=[
            'Gen', 'Max', 'Min', 'Std', 'Avg', 'Covered_demand', 'NCS'
        ])
        uncovered_demand = 1
        _max = -1
        # Begin the evolution
        while g < 1000 and ((uncovered_demand > 0) or (_max < 0)):
            init_time = time()
            # A new generation
            g += 1
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability 0.2
                if random.random() < 0.4:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability 0.2
                if random.random() < 0.4:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                try:
                    ind.fitness.values = (fit, )
                except TypeError:
                    ind.fitness.values = (fit[0], )

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            best_ind = tools.selBest(pop, 1)[0]
            uncovered_demand = unsatisfied_demand(best_ind)

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            _max = max(fits)

            print("  Min %s" % min(fits))
            print("  Max %s" % _max)
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("  Uncovered demand %s" % int(uncovered_demand))
            print("  Gen time: {:.3f}[s]".format(time() - init_time))
            print("  NCS %s" % int(x(best_ind).sum()))

            new_data = pd.DataFrame({
                'Gen': [g],
                'Max': [max(fits)],
                'Min': [min(fits)],
                'Std': [std],
                'Avg': [mean],
                'NCS': [x(best_ind).sum()],
                'Covered_demand': [y(best_ind).sum()]
            })

            out_df = pd.concat([out_df, new_data])
            out.update(pop)

        print("-- End of (successful) evolution --")
        out_df.to_csv(f'week_model/Sol_report({n}).csv')
        print(f'Solution report saved on week_model/Sol_report({n}).csv')
        with open(f'week_model/Sol({n}).npy', 'wb') as f:
            np.save(f, np.array(out))
        print(f'Solution saved on week_model/Sol({n}).npy')

        return out
    
    except KeyboardInterrupt:

        out_df.to_csv(f'week_model/Sol_report({n}).csv')
        print(f'Solution report saved on week_model/Sol_report({n}).csv')
        with open(f'week_model/Sol({n}).npy', 'wb') as f:
            np.save(f, np.array(out))
        print(f'Solution saved on week_model/Sol({n}).npy')

        return out


if __name__ == "__main__":
    main()