import os
import time
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient, Poly, Model, sum, equal_to, solve, decode_solution
from dwave.system import DWaveSampler, EmbeddingComposite
from ortools.constraint_solver import pywrapcp

def print_solution(x, n):
    """ Print solution matrix """
    for row in x:
        print(" ".join(f"{var.Value():2}" for var in row))
    print()


# OR-Tools Solver
def solve_ortools(n, max_solutions):
    solver = pywrapcp.Solver('Magic square')
    x = [[solver.IntVar(1, n * n, f'x[{i}, {j}]') for j in range(n)] for i in range(n)]
    x_flat = [var for row in x for var in row]
    s_value = n * (n * n + 1) // 2

    solver.Add(solver.AllDifferent(x_flat, True))
    for i in range(n):
        solver.Add(solver.Sum(x[i]) == s_value)
        solver.Add(solver.Sum([x[j][i] for j in range(n)]) == s_value)
    solver.Add(solver.Sum([x[i][i] for i in range(n)]) == s_value)
    solver.Add(solver.Sum([x[i][n - i - 1] for i in range(n)]) == s_value)

    db = solver.Phase(x_flat, solver.CHOOSE_MIN_SIZE_LOWEST_MAX, solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    start_time = time.time()
    count = 0
    while solver.NextSolution():
        count += 1
        print(f"=== Solution {count} ===")
        print_solution(x, n)
        if count >= max_solutions:
            break
    solver.EndSearch()
    return time.time() - start_time

def create_qubo(n):
    s = n * (n * n + 1) // 2  # magic constant
    bqm = Model()

    # generate variables
    x = np.array([[[Poly() for _ in range(n * n)] for _ in range(n)] for _ in range(n)])

    # one value per cell
    print("Adding constraint for each grid cell having a value...")
    for i in range(n):
        for j in range(n):
            bqm += equal_to(sum(x[i, j]), 1)

    # each number is used exactly once
    print("Adding constraints for unique number usage...")
    for v in range(n * n):
        bqm += equal_to(sum(x[i, j, v] for i in range(n) for j in range(n)), 1)

    # sum of each row is s
    print("Adding constraints for row sums...")
    for i in range(n):
        bqm += equal_to(sum((v + 1) * x[i, j, v] for j in range(n) for v in range(n * n)), s)

    # sum of each column is s
    print("Adding constraints for column sums...")
    for j in range(n):
        bqm += equal_to(sum((v + 1) * x[i, j, v] for i in range(n) for v in range(n * n)), s)

    # sum of each diagonal is s
    print("Adding constraints for diagonal sums...")
    bqm += equal_to(sum((v + 1) * x[i, i, v] for i in range(n) for v in range(n * n)), s)
    bqm += equal_to(sum((v + 1) * x[i, n - i - 1, v] for i in range(n) for v in range(n * n)), s)
    print("Number of variables in QUBO: ", bqm.num_variables)
    print("Number of constraints in QUBO: ", len(bqm))

    return bqm, x

# Amplify Solver
def solve_amplify(n, max_solutions):
    # Ensure D-Wave API token is set
    os.environ["AMPLIFY_API_TOKEN"] = "AE/EUKPYiVF6yYhSCIipu5eLbPq62MjqLYu"  # Set your Amplify API token

    client = FixstarsClient()
    client.token = os.getenv("AMPLIFY_API_TOKEN")

    model, x = create_qubo(n)

    start_time = time.time()
    result = solve(model, client)
    elapsed_time = time.time() - start_time

    if len(result) == 0:
        print("No solution found")
        return elapsed_time

    solutions = result[:max_solutions]

    for idx, sol in enumerate(solutions):
        print(f"=== Amplify Solution {idx + 1} ===")
        magic_square = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                for v in range(n * n):
                    if decode_solution(sol.values[x[i, j, v]]):
                        magic_square[i, j] = v + 1
        print(magic_square)
    
    return elapsed_time

# Benchmarking
def benchmark():
    ns = [4, 5, 6, 7]
    max_solutions_list = [1, 10, 100]

    results = {method: [] for method in ["OR-Tools", "MiniZinc", "Amplify", "D-Wave"]}
    for n in ns:
        for max_solutions in max_solutions_list:
            if n == 7 and max_solutions != 1:
                continue
            print(f"Running for n={n}, max_solutions={max_solutions}...")
            results["OR-Tools"].append(solve_ortools(n, max_solutions))
            results["Amplify"].append(solve_amplify(n, max_solutions))

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [f"{n}-{ms}" for n in ns for ms in max_solutions_list if not (n == 7 and ms != 1)]
    x = np.arange(len(x_labels))
    width = 0.2

    ax.bar(x - 1.5 * width, results["OR-Tools"], width, label="OR-Tools")
    ax.bar(x + 0.5 * width, results["Amplify"], width, label="Amplify")

    ax.set_xlabel("n - max_solutions")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Magic Square Solver Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig("benchmark.png")
    plt.show()


if __name__ == '__main__':
    benchmark()
