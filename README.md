# Strengths and Weaknesses of Genetic Algorithms

## Peer Evaluation Instructions
**Note**: These instructions were formulated for Mac and may not work for Windows users

1. Read the Objective and Background sections at the bottom of this README to familiarize yourself with our project motivations.
2. Download the .zip file to your computer.
3. Ensure you have a relatively recent version of Python installed (likely 3.7+). You can check your version by running the following command in your terminal:

`python3 --version`

5. Install a few packages if you have not already by running the following commands in the terminal:

`pip3 install numpy`

`pip3 install matplotlib`

3. Using the terminal, enter into the directory of the .zip folder and run the following command:

`python3 peer_eval.py`

4. Now, wait for the code to execute and you should see the output printed in the console (on my machine the entire execution took less than 2 minutes – but this will vary by machine).
5. For more detailed explanation of the code and results check out the Jupyter Notebook (.ipynb) for each specific problem (these notebooks contain code, explanation, output, and analysis).

---

## Objective
Compare the performance of genetic algorithms versus traditional approaches for three different types of problems.

## Background
1. Knapsack
  -  Problem description: Find the optimal packing of a knapsack with given weight capacity and a set of items with individual values and weights such that the value of the knapsack is maximized

     - https://www.educative.io/answers/what-is-the-knapsack-problem
  -  Traditional approach: **Recursive backtracking**

     -  https://www.tutorialspoint.com/introduction-to-backtracking
2. N-Queens
  - Problem description: Find a valid placement of N Queens on an NxN chessboard such that no Queen is under attack by any other Queen

    - https://en.wikipedia.org/wiki/Eight_queens_puzzle
  - Traditional approach: **Dynamic programming**

    - https://www.programiz.com/dsa/dynamic-programming
3. Class Schedule
  - Problem description: Find a valid class ordering given a list of courses with prerequisites (and possibly some other desired critieria)
  - Traditional approach: **Topological sort**

    - https://en.wikipedia.org/wiki/Topological_sorting
