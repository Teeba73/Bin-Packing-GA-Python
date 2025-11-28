"""
bin_packing.py

Single-file experiment script:
- Genetic Algorithm based on permutation encoding + First-Fit decoder
- Baseline heuristics: FF, FFD, BF, BFD
- Prints number of bins used, items in each bin, and GA best fitness
"""

import random

# -----------------------------
# Decoder & helper functions
# -----------------------------

def first_fit_decode(items, permutation, bin_capacity):
    bins = []
    bin_remaining = []
    assignment = [-1] * len(items)
    for idx in permutation:
        size = items[idx]
        placed = False
        for b_i, rem in enumerate(bin_remaining):
            if size <= rem:
                bins[b_i].append(idx)
                bin_remaining[b_i] -= size
                assignment[idx] = b_i
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(bin_capacity - size)
            assignment[idx] = len(bins) - 1
    return bins, bin_remaining, assignment

def evaluate_solution_bins(bins, bin_remaining, bin_capacity):
    used = len(bins)
    return used

# -----------------------------
# Heuristics
# -----------------------------

def first_fit(items, bin_capacity):
    bins = []
    bin_remaining = []
    assignment = [-1] * len(items)
    for i, size in enumerate(items):
        placed = False
        for b_i, rem in enumerate(bin_remaining):
            if size <= rem:
                bins[b_i].append(i)
                bin_remaining[b_i] -= size
                assignment[i] = b_i
                placed = True
                break
        if not placed:
            bins.append([i])
            bin_remaining.append(bin_capacity - size)
            assignment[i] = len(bins) - 1
    return bins, bin_remaining, assignment

def first_fit_decreasing(items, bin_capacity):
    order = sorted(range(len(items)), key=lambda i: -items[i])
    bins, bin_remaining, _ = first_fit([items[i] for i in order], bin_capacity)
    mapped_bins = []
    for b in bins:
        mapped_bins.append([order[idx] for idx in b])
    mapped_remaining = [bin_capacity - sum(items[idx] for idx in b) for b in mapped_bins]
    return mapped_bins, mapped_remaining

def best_fit(items, bin_capacity):
    bins = []
    bin_remaining = []
    assignment = [-1] * len(items)
    for i, size in enumerate(items):
        best_index = -1
        best_rem = None
        for b_i, rem in enumerate(bin_remaining):
            if size <= rem:
                if best_rem is None or rem - size < best_rem:
                    best_rem = rem - size
                    best_index = b_i
        if best_index >= 0:
            bins[best_index].append(i)
            bin_remaining[best_index] -= size
            assignment[i] = best_index
        else:
            bins.append([i])
            bin_remaining.append(bin_capacity - size)
            assignment[i] = len(bins) - 1
    return bins, bin_remaining, assignment

def best_fit_decreasing(items, bin_capacity):
    order = sorted(range(len(items)), key=lambda i: -items[i])
    sizes = [items[i] for i in order]
    bins, bin_remaining, _ = best_fit(sizes, bin_capacity)
    mapped_bins = []
    for b in bins:
        mapped_bins.append([order[idx] for idx in b])
    mapped_remaining = [bin_capacity - sum(items[idx] for idx in b) for b in mapped_bins]
    return mapped_bins, mapped_remaining

# -----------------------------
# Genetic Algorithm
# -----------------------------

def initialize_population(num_items, pop_size, rng):
    pop = []
    base = list(range(num_items))
    for _ in range(pop_size):
        individual = base.copy()
        rng.shuffle(individual)
        pop.append(individual)
    return pop

def fitness_of_individual(individual, items, bin_capacity):
    bins, bin_remaining, _ = first_fit_decode(items, individual, bin_capacity)
    used = evaluate_solution_bins(bins, bin_remaining, bin_capacity)
    return used  # lower is better

def tournament_selection(pop, fitnesses, tournament_size, rng):
    picked_indices = rng.choices(range(len(pop)), k=tournament_size)
    best = picked_indices[0]
    for idx in picked_indices[1:]:
        if fitnesses[idx] < fitnesses[best]:
            best = idx
    return best

def order_crossover(parent_a, parent_b, rng):
    n = len(parent_a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[i:j+1] = parent_a[i:j+1]
    pos = (j+1) % n
    for k in range(n):
        candidate = parent_b[(j+1+k) % n]
        if candidate not in child:
            child[pos] = candidate
            pos = (pos + 1) % n
    return child

def swap_mutation(individual, rng):
    a, b = rng.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]

def scramble_mutation(individual, rng):
    a, b = sorted(rng.sample(range(len(individual)), 2))
    sub = individual[a:b+1]
    rng.shuffle(sub)
    individual[a:b+1] = sub

def evolve_ga(items, bin_capacity,
              pop_size=100,
              generations=200,
              crossover_rate=0.9,
              mutation_rate=0.2,
              tournament_k=3,
              elitism=2,
              rng_seed=None,
              verbose=True):
    rng = random.Random(rng_seed)
    num_items = len(items)
    pop = initialize_population(num_items, pop_size, rng)
    fitnesses = [fitness_of_individual(ind, items, bin_capacity) for ind in pop]

    best_solution = None
    best_fitness = float('inf')

    for gen in range(1, generations + 1):
        new_pop = []
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
        for e in range(elitism):
            new_pop.append(pop[sorted_idx[e]].copy())
        while len(new_pop) < pop_size:
            p1_idx = tournament_selection(pop, fitnesses, tournament_k, rng)
            p2_idx = tournament_selection(pop, fitnesses, tournament_k, rng)
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            offspring = p1.copy()
            if rng.random() < crossover_rate:
                offspring = order_crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                if rng.random() < 0.8:
                    swap_mutation(offspring, rng)
                else:
                    scramble_mutation(offspring, rng)
            new_pop.append(offspring)

        pop = new_pop
        fitnesses = [fitness_of_individual(ind, items, bin_capacity) for ind in pop]
        gen_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        gen_best_f = fitnesses[gen_best_idx]

        if gen_best_f < best_fitness:
            best_fitness = gen_best_f
            best_solution = pop[gen_best_idx].copy()

        if verbose and gen % max(1, generations // 20) == 0:
            print(f"[GA] Gen {gen}: best fitness {gen_best_f}")

    # Final decode best
    bins, bin_remaining, _ = first_fit_decode(items, best_solution, bin_capacity)
    return best_solution, best_fitness, bins

# -----------------------------
# Demo using 100 items
# -----------------------------

if __name__ == '__main__':
    bin_capacity = 100.0
    items = [
        20,70,50,30,40,60,10,80,25,35,45,55,15,65,5,75,12,38,48,58,22,32,42,52,62,72,18,28,36,46,
        56,66,76,86,26,34,44,54,64,74,14,24,34,44,54,64,74,84,16,26,36,46,56,66,76,86,96,12,22,32,
        42,52,62,72,82,92,11,21,31,41,51,61,71,81,91,13,23,33,43,53,63,73,83,93,17,27,37,47,57,67,
        77,87,97,19,29,39,49,59,69,79,89,99,9,6,3,1
    ]

    ga_params = {
        'pop_size': 80,
        'generations': 120,
        'crossover_rate': 0.9,
        'mutation_rate': 0.25,
        'tournament_k': 3,
        'elitism': 2
    }

    print("=== Running Heuristics ===")
    for name, func in [('FF', first_fit), ('FFD', first_fit_decreasing), ('BF', best_fit), ('BFD', best_fit_decreasing)]:
        if 'D' in name:
            bins, _ = func(items, bin_capacity)
        else:
            bins, _, _ = func(items, bin_capacity)
        print(f"\n{name} - bins used: {len(bins)}")
        for i, b in enumerate(bins):
            print(f"  Bin {i+1}: {[items[idx] for idx in b]}")

    print("\n=== Running GA ===")
    best_solution, best_fitness, bins = evolve_ga(items, bin_capacity, **ga_params, verbose=True)
    print(f"\nGA - bins used: {len(bins)}, best fitness: {best_fitness}")
    for i, b in enumerate(bins):
        print(f"  Bin {i+1}: {[items[idx] for idx in b]}")
