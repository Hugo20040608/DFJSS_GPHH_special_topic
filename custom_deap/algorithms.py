import random
import numpy as np
from deap import tools
import time
from tqdm import tqdm
from math import ceil


def remove_duplicates(population):
    new_population = []
    temp_list = []
    for ind in population:
        if str(ind) not in temp_list:
            new_population.append(ind)
            temp_list.append(str(ind))
    return new_population

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

def GPHH_new(population, toolbox, cxpb, mutpb, ngen, n, stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

##---------------------- milkreo: random
    # set random seed
    rand_value = random.randint(1,300)
##----------------------

##----------------------
    # # Initialize selection accuracy
    # selection_accuracy_list = []
    # selection_accuracy_list_simple = []
##----------------------

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1), desc="Evolution Progress"):
        # Simple progresstion output
        print(f"Generation {gen}/{ngen} started...")

        # produce offsprings
        offspring = varAnd(population, toolbox, cxpb, mutpb)
        population_1 = population + offspring

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = np.array([[ind, rand_value] for ind in population_1 if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population_1 if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) 
        
        # Assign fitness values to individuals
        for ind, fit in zip(invalid_ind_1, fitnesses):
            ind.fitness.values = (fit,) if not isinstance(fit, tuple) else fit

##----------------------
        # # generate the phenotypic characterization of the individuals
        # invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_1], dtype=object)
        # invalid_ind_1 = [ind for ind in population_1]
        # decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        # decision_matrix = np.array(decision_matrix)
        # fitness_matrix = [ind.fitness.values for ind in population_1]

        # # Update the surrogate models (train the predictors)
        # KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        # MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        # DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        # SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
##----------------------

        # Select the next generation individuals based on the evaluation
        pop_size = len(population)
        population = toolbox.select(population_1, pop_size)

##----------------------
        # # produce offsprings
        # population_intermediate_to_add = []
        # for i in range(n):
        #     offspring = varAnd(population, toolbox, cxpb, mutpb)
        #     population_intermediate_to_add += offspring

        # # initialize the intermediate population
        # population_intermediate = population + population_intermediate_to_add


        # # Evaluate the individuals with an invalid fitness
        # invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate if not ind.fitness.valid], dtype=object)
        # invalid_ind_1 = [ind for ind in population_intermediate if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind_1, fitnesses):
        #     ind.fitness.values = fit

        # # Assign a number to each individual in the intermediate population
        # for i in range(len(population_intermediate)):
        #     population_intermediate[i].number = i

        # # generate the phenotypic characterization of the individuals in the intermediate population
        # invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate], dtype=object)
        # decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        # decision_matrix = np.array(decision_matrix)

        # # Create a duplicate of the intermediate population for the prediction
        # population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]

        # # Estimation of the individuals fitness
        # prediction_KNN = predict(KNN_model, decision_matrix)
        # prediction_MLP = predict(MLP_model, decision_matrix)
        # prediction_DT = predict(DT_model, decision_matrix)
        # prediction_SVM = predict(SVM_model, decision_matrix)

        # # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        # invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        # for ind, fit in zip(invalid_ind, prediction_KNN):
        #     ind.fitness.values = fit
        # # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        # invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        # for ind, fit in zip(invalid_ind, prediction_MLP):
        #     ind.fitness.values = fit
        # # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        # invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        # for ind, fit in zip(invalid_ind, prediction_DT):
        #     ind.fitness.values = fit
        # # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        # invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        # for ind, fit in zip(invalid_ind, prediction_SVM):
        #     ind.fitness.values = fit

        # # Select the next generation individuals based on the evaluation
        # pop_size = len(population)
        # population = toolbox.select(population_intermediate, pop_size)

        # # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        # population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        # population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        # population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        # population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)

        # # compare both selections and calculate the selection accuracy
        # ranking = []
        # for i in population:
        #     ranking.append(i.number)
        # print(ranking)

        # ranking_predicted_KNN = []
        # ranking_predicted_MLP = []
        # ranking_predicted_DT = []
        # ranking_predicted_SVM = []
        # for i in population_predicted_KNN:
        #     ranking_predicted_KNN.append(i.number)
        # for i in population_predicted_MLP:
        #     ranking_predicted_MLP.append(i.number)
        # for i in population_predicted_DT:
        #     ranking_predicted_DT.append(i.number)
        # for i in population_predicted_SVM:
        #     ranking_predicted_SVM.append(i.number)
        # print(ranking_predicted_KNN)
        # selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
        #                                                  pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        # selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
        #                                                  pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        # selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
        #                                                  pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        # selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
        #                                                  pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))

        # selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT, selection_accuracy_SVM])
        # selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple, selection_accuracy_SVM_simple])

        # print('selection accuracy ')
        # print(f'KNN: {selection_accuracy_list}')
        # print('selection accuracy simple')
        # print(f'KNN: {selection_accuracy_list_simple}')
##----------------------

        # Update the hall of fame with the generated individuals
        if halloffame is not None and len(halloffame) > 0:
            best = halloffame[0]
            print(f"Best individual at Generation {gen}: Fitness = {best.fitness.values[0]}")
        else:
            best = tools.selBest(population, 1)[0]
            print(f"Best individual at Generation {gen}: Fitness = {best.fitness.values[0]}")

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

##----------------------
    # return population, logbook, selection_accuracy_list, selection_accuracy_list_simple
##----------------------
    return population, logbook

def GPHH_experiment2_WS(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):

    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start = time.time()
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population))]
        print(f'population size of intermediate population: {len(population_intermediate)}')


        # add the selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end-start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time-start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict

