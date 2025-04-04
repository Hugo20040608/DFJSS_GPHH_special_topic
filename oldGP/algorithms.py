import random
import numpy as np
from deap import tools
import time
from tqdm import tqdm

def remove_duplicates(population):
    new_population = []
    temp_list = []
    for ind in population:
        if str(ind) not in temp_list:
            new_population.append(ind)
            temp_list.append(str(ind))
    return new_population

# def remove_duplicates(population):
#     new_population = []
#     for ind in population:
#         is_duplicate = False
#         for existing_ind in new_population:
#             if differ_in_one_leaf(ind, existing_ind):
#                 is_duplicate = True
#                 break
#         if not is_duplicate:
#             new_population.append(ind)
#     return new_population

# def differ_in_one_leaf(ind1, ind2):
#     # 確保兩個個體的長度相同
#     if len(ind1) != len(ind2):
#         return False
#     diff_count = 0
#     for node1, node2 in zip(ind1, ind2):
#         # 只比較葉節點
#         if node1.arity == 0 and node2.arity == 0:
#             if node1 != node2:
#                 diff_count += 1
#                 if diff_count > 1:
#                     return False
#         # 如果非葉節點不同，視為不同個體
#         elif node1 != node2:
#             return False
#     return diff_count == 1

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            offspring[i - 1].phenotypic = None
            offspring[i].phenotypic = None

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            offspring[i].phenotypic = None
    return offspring

def GPHH_new(population, toolbox, cxpb, mutpb, ngen, rand_seed, stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    random.seed(a=rand_seed)
    rand_value = random.randint(1,300)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Simple progresstion output
        # print(f" Generation {gen}/{ngen} started...")

        # produce offsprings
        random.shuffle(population)
        offspring = varAnd(population, toolbox, cxpb, mutpb)
        population_1 = population + offspring

        # Evaluate the individuals with an invalid fitness using simple evaluation
        invalid_ind = np.array([[ind, rand_value] for ind in population_1 if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population_1 if not ind.fitness.valid]

        # 計算 fitness (將 toolbox.evaluate 函數應用到 invalid_ind 列表中的每個元素上)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) 
        
        # Assign fitness values to individuals
        for ind, fit in zip(invalid_ind_1, fitnesses):
            # 將 fit 的型別轉換成 tuple
            ind.fitness.values = (fit,) if not isinstance(fit, tuple) else fit

        # remove duplicates 
        population_1 = remove_duplicates(population=population_1)

        # Select the next generation individuals based on the evaluation
        pop_size = len(population)
        population = toolbox.select(population_1, pop_size)

        # Update the hall of fame with the generated individuals
        if halloffame is not None and len(halloffame) > 0:
            best = halloffame[0]
            # print(f"Best individual at Generation {gen}: Fitness = {best.fitness.values[0]}")
        else:
            best = tools.selBest(population, 1)[0]
            # print(f"Best individual at Generation {gen}: Fitness = {best.fitness.values[0]}")

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# def GPHH_experiment2_WS(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
#              halloffame=None, verbose=__debug__):

#     # Initialize count for total time
#     start_total_time = time.time()

#     logbook = tools.Logbook()
#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

#     # set random seed
#     rand_value = random.randint(1,300)

#     # initialize the lists for measuring the time
#     time_per_generation = []

#     # Update random seed
#     rand_value = random.randint(1,300)

#     # Initialize lists for populations
#     population_dict = {}
#     fitness_dict = {}

#     # Initialize the population to train the classifier
#     pop_train = []
#     pop_size = int(len(population)/2)
#     pop_size_offspring = int(len(population))

#     # Begin the generational process
#     for gen in range(1, ngen + 1):
#         start = time.time()
#         print(f'Generation: {gen}')
#         population = remove_duplicates(population=population)
#         print(f'population size: {len(population)}')

#         # Evaluate the individuals with an invalid fitness using full evaluation
#         invalid_ind = [ind for ind in population if not ind.fitness.valid]
#         fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit
#         print(f'Evaluated individuals: {len(invalid_ind)}')

#         # save the full evaluated individuals to the population dict
#         population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

#         fitness_list = [str(ind.fitness) for ind in population]
#         tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
#         f1_list = [a_tuple[0] for a_tuple in tuple_list]
#         f2_list = [a_tuple[1] for a_tuple in tuple_list]
#         fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
#         fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

#         # Select the next generation individuals based on the evaluation
#         population = toolbox.select(population, pop_size)
#         print(f'population size after selection: {len(population)}')

#         # produce offsprings
#         population_size_intermediate = len(population)
#         population_intermediate = []
#         while len(population_intermediate) < population_size_intermediate:
#             offspring = varAnd(population, toolbox, cxpb, mutpb)
#             population_intermediate += offspring
#             population_intermediate = remove_duplicates(population=population_intermediate)
#             population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

#         # cut the intermediate population to its defined size
#         population_intermediate = population_intermediate[:(len(population))]
#         print(f'population size of intermediate population: {len(population_intermediate)}')


#         # add the selected individuals to the population
#         if gen < ngen:
#             population += population_intermediate

#         print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

#         # Update the hall of fame with the generated individuals
#         if halloffame is not None:
#             halloffame.update(offspring)

#         # Update random seed
#         rand_value = random.randint(1, 300)
#         end = time.time()
#         time_current_generation = end-start
#         time_per_generation.append(time_current_generation)
#         print(f'Execution time one loop: {time_current_generation}')

#     end_total_time = time.time()
#     total_time = end_total_time-start_total_time
#     time_dict = {'time per generation': time_per_generation, 'total time': total_time}
#     return population, logbook, time_dict, population_dict, fitness_dict

