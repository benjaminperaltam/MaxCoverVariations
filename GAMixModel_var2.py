"""
Excecute with command python -m scoop GAMixModel.py
Tho choose the number of workers python -m scoop -n 12 GAMixModel.py
must be in deap_env

individuals have the format:
ind = [[1, 1, 2], [2, 5, 20], [3, None, None], ...]
In this case, the vehicle 1 is assignated to the CS 1 in the time 2,
the vehicle 2 is assignated to the CS 5 in the time 20 and the vehicle 3
is not assignated to any CS.
"""

n = 7
inds_path = "Inputs/inds200_feasible.npy"
calculate_reachable_nodes = False

import random
import numpy as np
import pandas as pd
import pickle
import warnings

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
from demand import edges
from shapely.geometry import Point
import geopandas as gpd

nodes = pd.read_csv('Inputs/nodes_N5054.csv')
D = pd.read_csv('Inputs/D_N2527.csv')
Tp = np.load(f'Inputs/Tp.npy', allow_pickle=True)
D.O = D.O.apply(
    lambda x: Point(
        [float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]
    )
)
D.D = D.D.apply(
    lambda x: Point(
        [float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]
    )
)
nodes.Node = nodes.Node.apply(
    lambda x: Point(
        [float(i) for i in x.replace('(', '').replace(')', '').split(' ')[1:]]
    )
)
nodes = gpd.GeoDataFrame(nodes, crs="EPSG:32718", geometry='Node')
idxs = np.load(f'Index_samples/N2527.npy', allow_pickle=True)

edges_sample = edges.iloc[idxs].reset_index()

Tp = Tp[::2] + Tp[1::2]

demand_ = Demand(D.apply(lambda x: DemandNode(O=x.O,
                                              D=x.D,
                                              f=x.f,
                                              tol=x.tol),
                        axis=1).to_numpy())
I = 2 * len(demand_)
J = len(demand_)
T = len(Tp)
K = 3 * int(48 / T) * np.ones(I)
Nce = 165

if calculate_reachable_nodes:
    reachablemask = {j: demand_[j].reachable_nodes(nodes=nodes).index \
                     for j in range(J)}
    with open('Inputs/reachable_nodes.pkl', 'wb') as fp:
        pickle.dump(reachablemask, fp)
        print('dictionary saved successfully to file')

else:
    with open('Inputs/reachable_nodes.pkl', 'rb') as fp:
        reachablemask = pickle.load(fp)


creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("map", futures.map)

def y(individual):
    assignationTime = ~np.isnan(individual).max(axis=1)
    reachable_ = np.array([individual[j, 1] in reachablemask[j] for j in range(J)])
    return (assignationTime & reachable_).astype(int)

def x(individual):
    activeIndex = np.unique(individual[:, 1][~np.isnan(individual).max(axis=1)])
    activeIndex = activeIndex.astype(int)
    out = np.zeros(I)
    out[activeIndex] = 1
    return out

def objective_function(individual):
    return 1e4 * D.f.to_numpy() @ y(individual)

def capacity(individual):
    CS = np.arange(I)
    out = np.zeros((I, T))
    ind_ = individual.astype(int)
    for t in range(T):
        tindividual = individual[ind_[:, 2] == t]
        count = np.array([(tindividual[:, 1] == i).sum() for i in CS])
        out[:, t] = K - count
    return out

def nce(individual):
    return Nce - x(individual).sum()

def timeConstr(individual):
    time = np.arange(T)
    ind_ = individual.astype(int)
    count = np.array([(ind_[:, 2] == t).sum() for t in time])
    return len(demand_) * Tp * 1.1 - count, count - len(demand_) * Tp * .9

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

def unsatisfied_demand(individual):
    z = individual
    satisfied_demand = y(z)
    return len(satisfied_demand) - satisfied_demand.sum()

def customMultiFlip(individual, indpb):
    CSMask = np.random.choice(a=[True, False], size=J, p=[indpb, 1 - indpb])
    TimeMask = np.random.choice(a=[True, False], size=J, p=[indpb, 1 - indpb])

    for j in range(J):
        if CSMask[j]:
            individual[j] = np.random.choice(np.concatenate((reachablemask[j],
                                                             np.array([np.nan]))))
    
    individual[TimeMask, 2] = np.random.choice(np.concatenate((np.arange(T),
                                                               np.array([np.nan]))),
                                               size=sum(TimeMask))
    
    individual[np.isnan(individual[:, 1]), 2] = np.nan
    individual[np.isnan(individual[:, 2]), 1] = np.nan

    return individual,

def customTwoPoint(ind1, ind2):
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

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "rb") as pop_file:
        contents = np.load(pop_file)
    return pcls((ind_init(c)) for c in contents)

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
        out_df.to_csv(f'deap_v1/Sol_report({n}).csv')
        print(f'Solution report saved on deap_v1/Sol_report({n}).csv')
        with open(f'deap_v1/Sol({n}).npy', 'wb') as f:
            np.save(f, np.array(out))
        print(f'Solution saved on deap_v1/Sol({n}).npy')

        return out
    
    except KeyboardInterrupt:

        out_df.to_csv(f'deap_v1/Sol_report({n}).csv')
        print(f'Solution report saved on deap_v1/Sol_report({n}).csv')
        with open(f'deap_v1/Sol({n}).npy', 'wb') as f:
            np.save(f, np.array(out))
        print(f'Solution saved on deap_v1/Sol({n}).npy')

        return out


if __name__ == "__main__":
    main()