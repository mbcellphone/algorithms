#!/usr/bin/env python3

"""
Module Name: linear_programming.py
Description: Examples for Linear Programming Algorithms (Simplex Method using SciPy)
Author: Marvin Billings
Date: 12/02/2024
"""

from scipy.optimize import linprog

# Simplex Method Example

def solve_linear_program():
    """
    Solves a linear programming problem using the Simplex method.
    Maximize: z = 3x1 + 2x2
    Subject to:
        2x1 + x2 <= 20
        4x1 + 3x2 <= 42
        2x1 + 5x2 <= 30
        x1, x2 >= 0
    """
    # Coefficients for the objective function (note: linprog minimizes, so we use negative for maximization)
    c = [-3, -2]

    # Coefficients for the inequality constraints (Ax <= b)
    A = [
        [2, 1],
        [4, 3],
        [2, 5]
    ]
    b = [20, 42, 30]

    # Bounds for x1 and x2 (x1, x2 >= 0)
    x0_bounds = (0, None)
    x1_bounds = (0, None)

    # Solving the linear program using linprog
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='simplex')

    # Printing results
    if res.success:
        print("Optimal value:", -res.fun)
        print("Values of decision variables:", res.x)
    else:
        print("The linear program did not find an optimal solution.")

# Main execution
if __name__ == "__main__":
    # Test Linear Programming Solution using Simplex Method
    print("Testing Linear Programming Solution using Simplex Method")
    solve_linear_program()
    print()

