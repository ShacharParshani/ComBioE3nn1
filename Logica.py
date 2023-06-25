import copy
import random

from Generation import Generation
from Weights import Weights
from global_processes import NUM_WEIGHT

import math


def crossover(p1, p2):
    random_cut = random.choice(range(1, NUM_WEIGHT - 1))
    new_p = Weights()
    for i in range(random_cut):
        new_p.weights[i] = p1.weights[i]
    for i in range(random_cut,NUM_WEIGHT):
        new_p.weights[i] = p2.weights[i]
    return new_p


class Logica:
    def __init__(self, pCrossover, pMut, n):
        self.numCrossover = math.ceil(pCrossover * n)  # number of crossover
        self.numRep = n - self.numCrossover  # number of replication
        self.numMut = math.ceil(pMut * n)  # number of mutation
        self.n = n  # n- size of population
        self.current_gen = Generation(self.n)
        self.current_gen.create_first_generation()

    def save_solution(self, permutation):
        with open('wnet1.txt', 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for value in permutation.weights:
                file.write(f"{value}\n")


    def run(self):
        total_iteration = 0
        max_fitness = 0
        count_fitness_no_change = 0
        while max_fitness < 0.8:
            i = 0
            count_fitness_no_change = 0
            count_test_fitness_down = 0
            last_fitness = 0
            last_test_fitness = 0
            while count_fitness_no_change <= 10 and count_test_fitness_down <= 3:
                self.current_gen = self.new_generation()
                print(f"generation: {i}")
                max_fitness = 0
                maxp = None
                for p in self.current_gen.generation:
                    # print(p.weights)
                    # print("fitness: ", p.fitness)
                    fitness = p.fitness
                    if fitness > max_fitness:
                        max_fitness = fitness
                        maxp = p

                print(f"max fitness: {max_fitness}")
                i += 1
                total_iteration += 1
                if max_fitness == last_fitness:
                    count_fitness_no_change += 1
                else:
                    count_fitness_no_change = 0

                test_fitness = maxp.cal_test_fitness()
                print(f"test fitness: {test_fitness}")
                if test_fitness < last_test_fitness:
                    count_test_fitness_down += 1
                else:
                    count_test_fitness_down = 0

                last_fitness = max_fitness
                last_test_fitness = test_fitness
            self.current_gen = Generation(self.n)
            self.current_gen.create_first_generation()

        print(f"finished after {total_iteration} generation")
        self.save_solution(maxp)

    def new_generation(self):
        self.current_gen.order_by_fitness()
        new_gen = Generation(self.n)
        # replication
        for i in range(self.numRep):
            new_gen.generation.append(self.replication(self.current_gen.generation[i]))
        # cross over
        options = self.current_gen.generation
        fitnesses = [p.fitness for p in self.current_gen.generation]
        sum_fit = sum(fitnesses)
        probabilities = [fit / sum_fit for fit in fitnesses]
        for i in range(self.numCrossover):
            random_p = random.choices(options, probabilities, k=2)
            p_1 = random_p[0]
            p_2 = random_p[1]
            new_gen.generation.append(crossover(p_1, p_2))
        # mutations
        random_indexes = random.sample(range(self.n), self.numMut)
        for i in range(self.numMut):
            p = new_gen.generation[random_indexes[i]]
            new_gen.generation[random_indexes[i]] = self.mutation(p)

        for i in range(self.n):
            new_gen.generation[i].upgrade_fitness()

        return new_gen

    def replication(self, p):
        newInstance = copy.deepcopy(p)
        return newInstance

    def mutation(self, p):  # switch one weight
        random_index = random.randint(0, NUM_WEIGHT - 1)
        mut_p = self.replication(p)
        mut_p.weights[random_index] = random.uniform(-1, 1)
        return mut_p


