import numpy as np
import copy
from bingo_multi_stage.bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo_multi_stage.bingo.symbolic_regression \
    import ExplicitRegression, ExplicitTrainingData, AGraph, \
           AGraphGenerator, ComponentGenerator, AGraphCrossover, AGraphMutation

from bingo_multi_stage.bingo.evolutionary_algorithms.generalized_crowding \
    import GeneralizedCrowdingEA
from bingo_multi_stage.bingo.evaluation.multi_evaluation import Evaluation
from bingo_multi_stage.bingo.evolutionary_optimizers.island import Island
from bingo_multi_stage.bingo.stats.hall_of_fame import HallOfFame

from bingo_multi_stage.bingo.symbolic_regression.adaptive_bayes_fitness_function \
    import BayesFitnessFunction
from bingo_multi_stage.bingo.selection.bayes_crowding import BayesCrowding

from bingo_multi_stage.bingo.util.log import configure_logging

configure_logging("detailed")




def get_training_data():
    data = np.load('../noisy_data.npy')
    x = data[:,1].reshape((-1,1))
    y = data[:,0].reshape((-1,1))
    return ExplicitTrainingData(x, y)


def make_fitness_functions(training_data, mcmc_steps, num_particles,
                          phi_exponent, smc_steps, num_multistarts):
    reg = ExplicitRegression(training_data)
    clo = ContinuousLocalOptimization(reg, algorithm="lm")
    bff = BayesFitnessFunction(clo,
                               num_particles=num_particles,
                               phi_exponent=phi_exponent,
                               smc_steps=smc_steps,
                               mcmc_steps=mcmc_steps,
                               num_multistarts=num_multistarts)
    return [clo, bff]


def make_island(fitness_functions, population_size, stack_size, operators,
                crossover_prob, mutation_prob):

    # generation
    component_generator = ComponentGenerator(
            input_x_dimension=fitness_functions[0].training_data.x.shape[1])
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(stack_size, component_generator,
                                use_python=True,
                                use_simplification=True)

    # variation
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    # evaluation
    num_procs=8
    evaluation_phase = Evaluation(fitness_functions, interval=30e6,
                                redundant=True, multiprocess=num_procs)
    # selection
    selection_phase = BayesCrowding()

    # evolutionary algorithm
    ea = GeneralizedCrowdingEA(evaluation_phase, crossover, mutation,
                               crossover_prob, mutation_prob,
                               selection=selection_phase)

    # island
    hof = HallOfFame(10,
                     similarity_function=lambda x, y: np.array_equal(
                                                x._simplified_command_array,
                                                y._simplified_command_array))
    #hofs = [copy.deepcopy(hof), copy.deepcopy(hof)]

    island = Island(ea, generator, population_size=population_size,
                    hall_of_fame=hof)
    return island

def run():
    # DATA PARAMS
    TRAINING_DATA = get_training_data()

    # BFF PARAMS
    NUM_PARTICLES = 300
    NUM_SMC_STEPS = 25
    NUM_MCMC_STEPS = 20
    PHI_EXPONENT = 8
    NUM_MULTISTARTS = 8

    # ISLAND PARAMS
    POPULATION_SIZE = 150

    # AGRAPH PARAMS
    OPERATORS = ["+", "-", "*", "/", "pow", "sin"]
    STACK_SIZE = 14
    USE_SIMPLIFICATION = True

    # VARIATION PARAMS
    CROSSOVER_PROB = 0.4
    MUTATION_PROB = 0.4

    # EVOLUTION PARAMS
    NUM_GENERATIONS = 1000
    CHECK_FREQUENCY = 50 
    FITNESS_THRESHOLD = -np.inf

    CLO_BFF = make_fitness_functions(TRAINING_DATA, NUM_MCMC_STEPS, NUM_PARTICLES,
                                PHI_EXPONENT, NUM_SMC_STEPS, NUM_MULTISTARTS)
    ISLAND = make_island(CLO_BFF, POPULATION_SIZE, STACK_SIZE, OPERATORS,
                         CROSSOVER_PROB, MUTATION_PROB)

    ISLAND.evolve_until_convergence(max_generations=NUM_GENERATIONS, 
                                    fitness_threshold=FITNESS_THRESHOLD,
                                    convergence_check_frequency=CHECK_FREQUENCY,
                                    checkpoint_base_name='check')

    print("HOF")
    print(ISLAND.hall_of_fame)

if __name__ == "__main__":
    run()
