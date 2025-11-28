# Bin-Packing-GA-Python
This project implements and compares several approaches for solving the Bin Packing Problem (BPP) using both Genetic Algorithms and classical bin packing heuristics.
The goal is to minimize the number of bins needed to pack all items under a fixed capacity.

- Features
 Genetic Algorithm (GA)
Permutation-based encoding
First-Fit decoder
Tournament selection
Order Crossover (OX)
Swap & Scramble mutation
Elitism

Reports:
Best fitness
Total bins used
Items in each bin

- Baseline Heuristics
First Fit (FF)
First Fit Decreasing (FFD)
Best Fit (BF)
Best Fit Decreasing (BFD)

Each heuristic outputs:
Number of bins used
Assignment of items to bins

- Experiment Setup
Dataset: 100 items (user-provided dataset instead of random generation)
Capacity: User-defined bin capacity
No plotting involved

Console output includes:
Heuristic results
GA progress (best-of-generation)
GA final best solution

- How It Works
1. Heuristic Baselines
Each heuristic tries to place each item in a bin based on its rules.
These are fast and give a good benchmark.

2. Genetic Algorithm
GA tries to evolve a packing arrangement by:
Shuffling permutations of items
Using First-Fit to evaluate each permutation
Selecting the best individuals
Applying crossover + mutation
Iterating over multiple generations

3. Output
At the end, the script prints:
Best heuristic result
GA best result
Full bin assignments
Final best fitness

- Running the Script
python bin_packing_ga_experiment.py

Make sure you:
Add your dataset (list of 100 item sizes)
Set your bin capacity
Run the script

- Example Output (Console)
[FF] bins used: 12
[BFD] bins used: 10
[GA] Gen 20: best fitness 10.003100, bins_used=10
[GA] Final best solution:
Bins used: 9
Bin 0: [items...]
Bin 1: [items...]
...
Best Fitness: 9.001200

- What This Project Demonstrates
Practical usage of genetic algorithms for NP-hard problems
Benchmarking GA against classical heuristics
Clear, understandable output suitable for research or teaching
Clean modular structure for extending crossover/mutation operators

- Perfect for:

Optimization course projects
AI / Evolutionary Computation studies

Comparing search-based vs rule-based algorithms

LinkedIn portfolio posts
