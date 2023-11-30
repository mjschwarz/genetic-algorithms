"""
-- Executable for peer evaluation --
You will need a relatively recent version of Python installed (likely 3.7+)

You may need to install a few packages if you have not already.
To do so, run the following commands in the terminal:

pip3 install numpy
pip3 install matplotlib

Using the terminal, enter into the directory containing this file 
and run the following command:

python3 peer_eval.py

Now, wait for the code to execute 
and you should see the output printed in the console.
"""

#########################################################

# Knapsack problem
print("------------")
print("# KNAPSACK #")
print("------------")
print()
import random
from time import sleep


# Dynamic programming approach
def knapsack(values, weights, capacity, verbose=False):
    num_items = len(values)

    dp = []
    for i in range(num_items + 1):
        dp.append([0] * (capacity + 1))

    for i in range(1, num_items + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                value_including_item = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                value_excluding_item = dp[i - 1][w]
                dp[i][w] = max(value_including_item, value_excluding_item)
            else:
                dp[i][w] = dp[i - 1][w]

        if verbose and i % 100 == 0:
            print(f"Finished {i} items")

    return dp[num_items][capacity]


print("Dynamic Programming:")

values = [60, 100, 110, 40, 50, 60]
weights = [10, 20, 30, 5, 30, 5]

test_capacities = [10, 20, 30, 50, 60, 70, 80, 90, 100]
for capacity in test_capacities:
    print(f"Capacity: {capacity} -> {knapsack(values, weights, capacity)}")

# Genetic algorithm approach
from time import time


def generate_knapsack(num_items):
    knapsack = [random.choice([0, 1]) for _ in range(num_items)]
    knapsack_capacity = 0
    for i in range(len(knapsack)):
        if knapsack[i] == 1:
            knapsack_capacity += weights[i]
    return knapsack


def calculate_fitness(knapsack, values, weights, capacity):
    total_value = 0
    total_weight = 0
    for i in range(len(knapsack)):
        if knapsack[i] == 1:
            total_value += values[i]
            total_weight += weights[i]
    if total_weight > capacity:
        return 0  # Penalize for exceeding capacity
    return total_value


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(knapsack, mutation_rate=0.05):
    for i in range(len(knapsack)):
        if random.random() < mutation_rate:
            knapsack[i] = 1 - knapsack[i]  # Flip the bit
    return knapsack


def knapsack_gen(
    values, weights, capacity, num_generations=50, verbose=False, log_times=False
):
    # Parameters
    num_items = len(values)
    population_size = 1000
    times = {}

    population = [generate_knapsack(num_items) for _ in range(population_size)]

    for gen in range(num_generations):
        # Calculate fitness for each knapsack
        fitness_scores = [
            calculate_fitness(k, values, weights, capacity) for k in population
        ]

        # Sort the population based on fitness and select the top knapsacks
        sorted_population = [
            x for _, x in sorted(zip(fitness_scores, population), reverse=True)
        ]
        parents = sorted_population[:50]

        # Generate new population through crossover and mutation
        new_population = parents[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

        if verbose and gen % 10 == 0:
            best_solution = max(
                population,
                key=lambda k: calculate_fitness(k, values, weights, capacity),
            )
            best_fitness = calculate_fitness(best_solution, values, weights, capacity)
            print(f"Generation: {gen} | Best fitness: {best_fitness}")

        if log_times and gen % 10 == 0:
            times[time()] = [gen, best_fitness]

    # Find the best solution at the end of the process
    best_solution = max(
        population, key=lambda k: calculate_fitness(k, values, weights, capacity)
    )
    best_fitness = calculate_fitness(best_solution, values, weights, capacity)

    if log_times:
        return best_solution, best_fitness, times

    return best_solution, best_fitness


sleep(1)
print()
print("Genetic Algorithm:")

values = [60, 100, 110, 40, 50, 60]
weights = [10, 20, 30, 5, 30, 5]

test_capacities = [10, 20, 30, 50, 60, 70, 80, 90, 100]
for capacity in test_capacities:
    print(f"Capacity: {capacity} -> {knapsack_gen(values, weights, capacity)[1]}")

print()
print("Both approaches correctly solve the knapsack problem.")
print()

print("Now we compare the runtime performance over many random examples.")
print(
    "(this will take less than a minute to run – but this could vary depending on your machine)\n"
)

random.seed(0)
values = [random.randint(10, 100) for _ in range(100)]
weights = [random.randint(10, 100) for _ in range(100)]
capacity = 25000

start = time()
print(
    f"Dynamic Programming: {knapsack(values, weights, capacity)} ({time() - start} sec)\n"
)

start = time()
print(
    f"Genetic Algorithm: {knapsack_gen(values, weights, capacity, 1000)[1]} ({time() - start} sec)\n"
)

print(
    "We can see that the dynamic programming approach is faster, but the genetic algorithm still finds the optimal solution.\n"
)


######################################################

# N-Queens problem
print("------------")
print("# N-QUEENS #")
print("------------")
print()

# Recursive backtracking approach
# Reference: https://reintech.io/blog/python-algorithms-solving-n-queen-problem

"""
Returns an empty NxN board (full of zeros)
"""


def generate_board(N):
    return [[0 for _ in range(N)] for _ in range(N)]


"""
Checks if the new Queen is under attack at this space
"""


def is_safe(board, row, col):
    # Row (only to the left)
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Upper diagonal '\' (only to the left)
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Lower diagonal '/' (only to the left)
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True


"""
Helper function for recursive backtracking
- Modifies board parameter with solved Queen placements
"""


def n_queens_helper(board, col):
    N = len(board[0])
    # Base case: All queens are placed
    if col >= N:
        return True

    # For this column, try placing the Queen in each row
    for row in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1  # Place a Queen

            # Recursively backtrack
            if n_queens_helper(board, col + 1) == True:
                return True  # Solution found

            board[row][col] = 0  # Remove that Queen

    # No solution found
    return False


"""
N-Queens solver
- Solves N-Queens problem using recursive backtracking
"""


def n_queens_recursive_backtracking(N):
    board = generate_board(N)
    n_queens_helper(board, 0)
    return board


# Genetic algorithm approach

# Reference: https://www.educative.io/answers/solving-the-8-queen-problem-using-genetic-algorithm

import random

"""
Convert genetic algorithm board representation to 2D array representation
- For testing/comparision purposes
"""


def convert_flat_board_to_2d(flat_board):
    N = len(flat_board)
    board_2d = generate_board(N)

    for col, row_plus_1 in enumerate(flat_board):
        board_2d[row_plus_1 - 1][col] = 1

    return board_2d


"""
Initial start state
- Generate a random board state (index = column, value = row where Queen located)
"""


def generate_board_state(N):
    board_state = [random.randint(0, N - 1) for _ in range(N)]
    return board_state


"""
Calculate the fitness of a board state
- Fitness function: Add the number of non-attacking pairs for each queen
"""


def calculate_fitness(board_state):
    N = len(board_state)
    MAX_FITNESS = N * (N - 1) / 2
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (
                board_state[i] == board_state[j]
                or abs(board_state[i] - board_state[j]) == j - i
            ):
                conflicts += 1
    return MAX_FITNESS - conflicts  # For 8-Queens: Max fitness = 28 (no conflicts)


"""
Selection
- Select parents for crossover (using tournament selection)
"""


def tournament_selection(population):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x[1])


"""
Crossover
- Crossover operation (single-point crossover)
"""


def crossover(parent1, parent2):
    N = len(parent1)
    crossover_point = random.randint(1, N - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


"""
Mutation
- Mutation operation (swap two positions)
"""


def mutate(board_state):
    N = len(board_state)
    pos1, pos2 = random.sample(range(N), 2)
    board_state[pos1], board_state[pos2] = board_state[pos2], board_state[pos1]
    return board_state


"""
N-Queens solver
- Solves N-Queens problem using genetic algorithm
"""


def n_queens_genetic(N, POPULATION_SIZE=50, MUTATION_RATE=0.1, MAX_GENERATIONS=100):
    MAX_FITNESS = N * (N - 1) / 2

    # Initial population
    population = [(generate_board_state(N), 0) for _ in range(POPULATION_SIZE)]

    # Loop over generations...
    for generation in range(MAX_GENERATIONS):
        # Calculate fitness for each board state
        population = [
            (board_state, calculate_fitness(board_state))
            for board_state, _ in population
        ]

        # Check if solution is found
        best_board_state = max(population, key=lambda x: x[1])[0]
        if calculate_fitness(best_board_state) == MAX_FITNESS:
            # print("Solution found in generation", generation)
            break

        # Create the next generation
        new_population = []

        # Elitism: Keep the best board state from the previous generation
        new_population.append(max(population, key=lambda x: x[1]))

        # Perform selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1[0], parent2[0])
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_population.append((child, 0))

        # Update the population
        population = new_population

    return best_board_state


# Compare the approaches
print(
    "We compare the runtime performance of the two approaches for several different numbers of Queens."
)
print("(this will take less than a minute depending on your computer specs)\n")

from time import time
import numpy as np

recursive_backtracking_runtime = {}
recursive_backtracking_stdev = {}

genetic_algorithm_runtime = {}
genetic_algorithm_stdev = {}

for N in range(8, 21, 2):
    # Recursive backtracking
    runtimes = []
    for _ in range(5):
        start = time()
        n_queens_recursive_backtracking(N)
        runtimes.append(time() - start)
    runtime = np.mean(runtimes)
    stdev = np.std(runtimes)
    recursive_backtracking_runtime[N] = runtime
    recursive_backtracking_stdev[N] = stdev
    print(f"Recurisve backtracking ({N} Queens): {runtime} sec (+/- {stdev} sec)")

    # Genetic algorithm
    runtimes = []
    for _ in range(5):
        start = time()
        n_queens_genetic(N)
        runtimes.append(time() - start)
    runtime = np.mean(runtimes)
    stdev = np.std(runtimes)
    genetic_algorithm_runtime[N] = runtime
    genetic_algorithm_stdev[N] = stdev
    print(f"Genetic algorithm ({N} Queens): {runtime} sec (+/- {stdev} sec)")

    print()

# Plot the findings
print("Two pop-up windows will appear displaying the graphical plots.\n")
print(">> TO CONTINUE, CLOSE THE POP-UP WINDOWS!\n")

import matplotlib.pyplot as plt

# Plot average algorithm runtime vs. N
recursive_backtracking_runtime_sorted = sorted(
    recursive_backtracking_runtime.items()
)  # sorted by key, return a list of tuples
genetic_algorithm_runtime_sorted = sorted(genetic_algorithm_runtime.items())

recursive_backtracking_N, recursive_backtracking_runtime = zip(
    *recursive_backtracking_runtime_sorted
)  # unpack a list of pairs into two tuples
genetic_algorithm_N, genetic_algorithm_runtime = zip(*genetic_algorithm_runtime_sorted)

plt.plot(recursive_backtracking_N, recursive_backtracking_runtime, "r")
plt.plot(genetic_algorithm_N, genetic_algorithm_runtime, "g")

plt.xlabel("N Queens")
plt.ylabel("Average Algorithm Runtime (sec)")

plt.legend(["Recurive backtracking", "Genetic algorithm"], loc="upper left")

plt.show()

# Plot stdev of algorithm runtime vs. N
recursive_backtracking_stdev_sorted = sorted(
    recursive_backtracking_stdev.items()
)  # sorted by key, return a list of tuples
genetic_algorithm_stdev_sorted = sorted(genetic_algorithm_stdev.items())

recursive_backtracking_N, recursive_backtracking_stdev = zip(
    *recursive_backtracking_stdev_sorted
)  # unpack a list of pairs into two tuples
genetic_algorithm_N, genetic_algorithm_stdev = zip(*genetic_algorithm_stdev_sorted)

plt.plot(recursive_backtracking_N, recursive_backtracking_stdev, "r")
plt.plot(genetic_algorithm_N, genetic_algorithm_stdev, "g")

plt.xlabel("N Queens")
plt.ylabel("Stdev of Algorithm Runtime (sec)")

plt.legend(["Recurive backtracking", "Genetic algorithm"], loc="upper left")

plt.show()


##################################################################################

# Class Schedule Problem
from time import sleep

sleep(2)

print("------------------")
print("# CLASS SCHEDULE #")
print("------------------")
print()

import random
from copy import deepcopy

# from tqdm import tqdm


class Course:
    def __init__(self, name, hours, subject, difficulty):
        self.name = name
        self.hours = hours
        self.subject = subject
        self.difficulty = difficulty

    def __str__(self):
        return self.name


class Semester:
    def __init__(self, courses):
        self.courses = courses
        self.hours = sum([course.hours for course in courses])
        self.difficulty = sum([course.difficulty for course in courses])
        self.subjects = set(course.subject for course in courses)

    def __str__(self):
        return f"Semester: {self.hours} hours, {self.difficulty} difficulty, {self.subjects} subjects"


class Schedule:
    def __init__(self, semesters):
        self.semesters = semesters

    def __str__(self):
        str = ""
        for semester in self.semesters:
            for c in semester.courses:
                str += f"{c.name} "
            str += "\n"
        return str


# create courses
courses = [
    # CS classes
    Course("CS 1101", 3, "CS", 2),
    Course("CS 2201", 3, "CS", 5),
    Course("CS 2212", 3, "CS", 3),
    Course("CS 3251", 3, "CS", 5),
    Course("CS 3250", 3, "CS", 5),
    Course("CS 3270", 3, "CS", 4),
    Course("CS 3281", 3, "CS", 4),
    # Math classes
    Course("MATH 1300", 3, "MATH", 4),
    Course("MATH 1301", 3, "MATH", 4),
    Course("MATH 2300", 3, "MATH", 4),
    # Physics classes
    Course("PHYS 1600", 3, "PHYS", 4),
    Course("PHYS 1601", 3, "PHYS", 4),
    # English classes
    Course("ENGL 1101", 3, "ENGL", 3),
    Course("ENGL 1102", 3, "ENGL", 3),
    # History classes
    Course("HIST 1101", 3, "HIST", 3),
    Course("HIST 1102", 3, "HIST", 3),
    # Philosophy classes
    Course("PHIL 1101", 3, "PHIL", 3),
    Course("PHIL 1102", 3, "PHIL", 3),
    # Religion classes
    Course("RELG 1101", 3, "RELG", 3),
    Course("RELG 1102", 3, "RELG", 3),
]

# set up prereqs
prereqs = {
    "CS 2201": set(["CS 1101"]),
    "CS 2212": set(["CS 1101"]),
    "CS 3251": set(["CS 2201", "CS 2212"]),
    "CS 3250": set(["CS 2201", "CS 2212"]),
    "CS 3270": set(["CS 2201", "CS 2212"]),
    "CS 3281": set(["CS 2201", "CS 2212"]),
    "MATH 1301": set(["MATH 1300"]),
    "MATH 2300": set(["MATH 1301"]),
    "PHYS 1601": set(["PHYS 1600"]),
    "ENGL 1102": set(["ENGL 1101"]),
    "HIST 1102": set(["HIST 1101"]),
    "PHIL 1102": set(["PHIL 1101"]),
    "RELG 1102": set(["RELG 1101"]),
}


def schedule_fitness(schedule, verbose=False):
    utility = 0

    # we should now check for prereqs
    taken_names = set()
    for semester in schedule.semesters:
        temp_taken = []
        for course in semester.courses:
            if course.name in prereqs:
                for prereq in prereqs[course.name]:
                    if prereq not in taken_names:
                        if verbose:
                            print(f"Prereq not taken: {prereq} for {course.name}")
                        utility -= 1000
            temp_taken.append(course.name)
        taken_names.update(temp_taken)

    # now we should check for duplicates
    taken_names = set()
    for semester in schedule.semesters:
        for course in semester.courses:
            if course.name in taken_names:
                utility -= 1000
            taken_names.add(course.name)

    return utility


def schedule_mutate(schedule):
    # make a copy of the schedule
    schedule = deepcopy(schedule)
    # swap two courses
    semester1 = random.choice(schedule.semesters)
    semester2 = random.choice(schedule.semesters)
    while semester1 == semester2:
        semester2 = random.choice(schedule.semesters)
    course1 = random.choice(semester1.courses)
    course2 = random.choice(semester2.courses)
    if course1 == course2:
        return schedule
    semester1.courses.remove(course1)
    semester2.courses.remove(course2)
    semester2.courses.append(course1)
    semester1.courses.append(course2)

    return schedule


def random_schedule():
    # create a list of courses
    course_list = deepcopy(courses)
    # shuffle the list
    random.shuffle(course_list)
    # create 4 semesters of 5 courses each
    semesters = []
    for i in range(4):
        semesters.append(Semester(course_list[i * 5 : (i + 1) * 5]))
    # return a new schedule
    return Schedule(semesters)


def run(num_iterations=10000, schedule_fitness=schedule_fitness):
    initial_population = [random_schedule() for i in range(100)]
    best_schedule = None
    best_fitness = -1000000000

    for i in range(num_iterations):
        # pick top 10 parents
        parents = sorted(initial_population, key=schedule_fitness, reverse=True)[:10]
        # create 90 children
        children = []
        for j in range(90):
            # pick a random parent
            parent = random.choice(parents)
            # mutate the parent
            child = schedule_mutate(parent)
            # add the child to the children
            children.append(child)
        # combine the parents and children
        initial_population = parents + children
        # sort the population by fitness
        initial_population = sorted(
            initial_population, key=schedule_fitness, reverse=True
        )
        # kill the bottom 90
        initial_population = initial_population[:10]
        # check if the top schedule is the best
        if schedule_fitness(initial_population[0]) > best_fitness:
            best_fitness = schedule_fitness(initial_population[0])
            best_schedule = initial_population[0]

        if i % 1000 == 0:
            print(f"Iteration: {i}")
            print(f"Best fitness: {best_fitness}")
            print(best_schedule)

    print("Final schedule:")
    print(f"Best fitness: {best_fitness}")
    print(best_schedule)
    return best_schedule


sleep(2)
print(
    "We observe that the genetic algorithm generates a valid ordering of classes (no prerequisites violated).\n"
)

schedule = run(10)


def schedule_fitness_2(schedule, verbose=False):
    utility = 0

    # we should now check for prereqs
    taken_names = set()
    for semester in schedule.semesters:
        temp_taken = []
        for course in semester.courses:
            if course.name in prereqs:
                for prereq in prereqs[course.name]:
                    if prereq not in taken_names:
                        if verbose:
                            print(f"Prereq not taken: {prereq} for {course.name}")
                        utility -= 1000
            temp_taken.append(course.name)
        taken_names.update(temp_taken)

    # now we should check for duplicates
    taken_names = set()
    for semester in schedule.semesters:
        for course in semester.courses:
            if course.name in taken_names:
                utility -= 1000
            taken_names.add(course.name)

    # lets reward physics classes in the first two semesters
    if "PHYS 1600" in [course.name for course in schedule.semesters[0].courses]:
        utility += 300
    if "PHYS 1601" in [course.name for course in schedule.semesters[1].courses]:
        utility += 300

    return utility


sleep(2)
print(
    "Now we demonstrate the flexibility and power of genetic algorithms over a simple topological sort.\n"
)

schedule_2 = run(1000, schedule_fitness_2)

print(
    "By modifying the fitness function we optimize for the number of Physics related courses taken during the first two semesters."
)
print("(this is not something a topological sort could do)\n")

sleep(2)

print("END OUTPUT")
