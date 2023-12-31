# Strengths and Weaknesses of Genetic Algorithms

## Peer Evaluation Instructions
`Note: These instructions were formulated for Mac and may need modification for Windows users.`

1. Read the *Objective* and *Background* sections at the bottom of this README to familiarize yourself with our project motivations.
2. Clone this repository to your computer.
3. Ensure you have a relatively recent version of Python installed (likely 3.7+). You can check your version by running the following command in your terminal:

```
python3 --version
```

5. Install a few packages if you have not already by running the following commands in the terminal:

```
pip3 install numpy
```

```
pip3 install matplotlib
```

3. Using the terminal, enter into the directory of the repository and run the following command:

```
python3 peer_eval.py
```

4. Now, wait for the code to execute and you should see the output printed in the console (on my machine the entire execution took less than 2 minutes – but this will vary by machine).
5. For more detailed explanation of the code and results check out the respecitve Jupyter Notebooks (.ipynb) for each specific problem.
- Jupyter Notebooks can be found in the `notebooks` directory (these notebooks contain code, explanation, output, and analysis).
- Project proposal, presentation, and report can be found in the `pdf` directory.

---

## Objective
Compare the performance of genetic algorithms versus traditional approaches for three different types of problems.

## Background
`For further reading, click on the external resource links.`
1. Knapsack
  -  [Problem description](https://www.educative.io/answers/what-is-the-knapsack-problem): Find the optimal packing of a knapsack with given weight capacity and a set of items with individual values and weights such that the value of the knapsack is maximized
  -  [Traditional approach](https://www.tutorialspoint.com/introduction-to-backtracking): Recursive backtracking

2. N-Queens
  - [Problem description](https://en.wikipedia.org/wiki/Eight_queens_puzzle): Find a valid placement of N Queens on an NxN chessboard such that no Queen is under attack by any other Queen
  - [Traditional approach](https://www.programiz.com/dsa/dynamic-programming): Dynamic programming

3. Class Schedule
  - Problem description: Find a valid class ordering given a list of courses with prerequisites (and possibly some other desired critieria)
  - [Traditional approach](https://en.wikipedia.org/wiki/Topological_sorting): Topological sort
