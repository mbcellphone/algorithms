#!/usr/bin/env python3

"""
Module Name: dynamic_programming.py
Description: Examples for Dynamic Programming Algorithms (Fibonacci Sequence, Longest Common Subsequence, Knapsack Problem)
Author: Marvin Billings
Date: 12/02/2024
"""

# Fibonacci Sequence using Dynamic Programming
def fibonacci(n):
    """
    Calculates the n-th Fibonacci number using dynamic programming.
    :param n: Position in the Fibonacci sequence.
    :return: n-th Fibonacci number.
    """
    if n <= 1:
        return n
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

# Longest Common Subsequence (LCS) using Dynamic Programming
def lcs(X, Y):
    """
    Finds the length of the longest common subsequence of two strings.
    :param X: First string.
    :param Y: Second string.
    :return: Length of LCS.
    """
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

# 0/1 Knapsack Problem using Dynamic Programming
def knapsack(W, wt, val, n):
    """
    Solves the 0/1 knapsack problem using dynamic programming.
    :param W: Maximum weight capacity of the knapsack.
    :param wt: List of item weights.
    :param val: List of item values.
    :param n: Number of items.
    :return: Maximum value that can be put in a knapsack of capacity W.
    """
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    
    return K[n][W]

# Main execution
if __name__ == "__main__":
    # Test Fibonacci Sequence
    print("Testing Fibonacci Sequence")
    n = 10
    print(f"Fibonacci number at position {n}: {fibonacci(n)}")
    print()

    # Test Longest Common Subsequence (LCS)
    print("Testing Longest Common Subsequence (LCS)")
    X = "AGGTAB"
    Y = "GXTXAYB"
    print(f"Length of LCS of '{X}' and '{Y}': {lcs(X, Y)}")
    print()

    # Test 0/1 Knapsack Problem
    print("Testing 0/1 Knapsack Problem")
    val = [60, 100, 120]
    wt = [10, 20, 30]
    W = 50
    n = len(val)
    print(f"Maximum value in knapsack of capacity {W}: {knapsack(W, wt, val, n)}")
    print()

