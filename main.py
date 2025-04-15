import math
import numpy as np
import bisect

class MaxFunctie:
    def __init__(self, population_size, left, right, a, b, c, precision, crossover_prob, mutation_prob, generations_number):
        self.population_size = population_size
        self.left = left
        self.right = right
        self.a, self.b, self.c = a, b, c
        self.p = precision
        self.crossover_prob = crossover_prob / 100
        self.mutation_prob = mutation_prob / 100
        self.generations_number = generations_number

        self.bit_number = math.ceil(abs(math.log2((right - left) * (10 ** precision))))
        self.max_value = float("-inf")
        self.disc = (right - left) / (2 ** self.bit_number)
        self.population = self._init_population()

    def _init_population(self):
        bits = [str(int(x > 0.5)) for x in np.random.rand(self.population_size * self.bit_number)]
        return [
            "".join(bits[i * self.bit_number: (i + 1) * self.bit_number])
            for i in range(self.population_size)
        ]

    def dec_to_bin(self, x):
        return bin(x)[2:] or "0"

    def codificare(self, x):
        k = self.dec_to_bin(int((x - self.left) // self.disc))
        return k.zfill(self.bit_number)

    def decodificare(self, x):
        return self.left + int(x, 2) * self.disc

    def fitness(self, x):
        return self.a * x * x + self.b * x + self.c

    def get_total_performance(self):
        return sum(self.fitness(self.decodificare(ch)) for ch in self.population)

    def get_selection_probability(self, chromosome):
        return self.fitness(self.decodificare(chromosome)) / self.get_total_performance()

    def _evolve_helper(self, log_file, print_info=False):
        def log(msg):
            if print_info:
                log_file.write(msg)

        # Populația inițială
        log("Populatia initiala\n")
        for i, chrom in enumerate(self.population):
            x = self.decodificare(chrom)
            f = self.fitness(x)
            log(f"{i + 1}: {chrom} x={x} f={f}\n")
        log("\n")

        # Selectie
        selection_probs = [self.get_selection_probability(ch) for ch in self.population]
        cumulative_prob = [0]
        best_idx = 0

        for i, p in enumerate(selection_probs):
            cumulative_prob.append(cumulative_prob[-1] + p)
            if p > selection_probs[best_idx]:
                best_idx = i

        log("Intervale probabilitati selectie\n" + str([float(x) for x in cumulative_prob]) + "\n")
        log(f"Cel mai bun cromozom este {best_idx + 1}\n")

        selectii = [self.population[best_idx]]
        for u in np.random.rand(self.population_size - 1):
            idx = bisect.bisect_left(cumulative_prob, u) - 1
            log(f"u = {u} selectam cromozomul {idx}\n")
            selectii.append(self.population[idx])
        log("\n")

        log("Dupa selectie:\n")
        for i, chrom in enumerate(selectii):
            x = self.decodificare(chrom)
            f = self.fitness(x)
            log(f"{i + 1}: {chrom} x={x} f={f}\n")
        log("\n")

        # Incrucișare
        def crossover(ch1, ch2, index):
            log(f"{ch1} {ch2} punct {index}\n")
            result = (ch1[:index] + ch2[index:], ch2[:index] + ch1[index:])
            log(f"rezultat {result[0]} {result[1]}\n")
            return result

        log(f"Probabilitatea de incrucisare {self.crossover_prob}\n")
        cross_picks = np.random.rand(self.population_size)
        crossover_points = np.random.randint(0, self.bit_number, size=self.population_size)
        idx1 = idx2 = None

        for i in range(self.population_size):
            if cross_picks[i] < self.crossover_prob:
                log(f"{i + 1}: {selectii[i]} u={cross_picks[i]} < {self.crossover_prob} participa\n")
                if idx1 is None:
                    idx1 = i
                else:
                    idx2 = i
                    log(f"recombinare dintre cromozomul {idx1} si cromozomul {idx2}:\n")
                    selectii[idx1], selectii[idx2] = crossover(selectii[idx1], selectii[idx2], crossover_points[idx1])
                    idx1 = idx2 = None
            else:
                log(f"{i + 1}: {selectii[i]} u={cross_picks[i]}\n")

        log("\nDupa recombinare:\n")
        for i, chrom in enumerate(selectii):
            x = self.decodificare(chrom)
            f = self.fitness(x)
            log(f"{i + 1}: {chrom} x={x} f={f}\n")
        log("\n")

        # Mutație
        log(f"Probabilitatea de mutatie pentru fiecare gena {self.mutation_prob}\n")
        mutation_flags = np.random.rand(self.population_size * self.bit_number)

        for i, chrom in enumerate(selectii):
            mutated = list(chrom)
            mutated_flag = False
            for j in range(self.bit_number):
                if mutation_flags[i * self.bit_number + j] < self.mutation_prob:
                    mutated[j] = '0' if mutated[j] == '1' else '1'
                    mutated_flag = True
            if mutated_flag:
                log(f"A fost modificat cromozomul {i + 1}\n")
                selectii[i] = "".join(mutated)

        # Evaluare
        self.fitness_sum = 0
        for i, chrom in enumerate(selectii):
            x = self.decodificare(chrom)
            f = self.fitness(x)
            self.fitness_sum += f
            if f > self.max_value:
                self.max_value = f
            log(f"{i + 1}: {chrom} x={x} f={f}\n")
        log("\n")

        self.population = selectii

    def evolve(self):
        with open("evolutie.txt", "w") as log:
            self._evolve_helper(log, print_info=True)
            log.write("Evolutia maximului:\n")
            log.write(f"Max fitness: {self.max_value}\n")
            log.write(f"Fitness Mean: {self.fitness_sum / self.population_size}\n\n")

            for _ in range(self.generations_number - 1):
                self._evolve_helper(log)
                log.write(f"Max fitness: {self.max_value}\n")
                log.write(f"Fitness Mean: {self.fitness_sum / self.population_size}\n\n")



alg = MaxFunctie(3, -10, 10, 1, 0, 0, 4, 80, 1, 100)
alg.evolve()

print("Maxim:", alg.max_value)
