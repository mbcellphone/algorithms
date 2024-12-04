#!/usr/bin/env python3

"""
Module Name: miscellaneous_algorithms.py
Description: Examples for Miscellaneous Algorithms (Union-Find/Disjoint Set, Fisher-Yates Shuffle)
Author: Marvin Billings
Date: 12/02/2024
"""

import random

# Union-Find (Disjoint Set) Implementation
class DisjointSet:
    def __init__(self, n):
        """
        Initializes the Disjoint Set with n elements.
        :param n: Number of elements.
        """
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, u):
        """
        Finds the representative of the set containing u.
        Uses path compression to speed up future queries.
        :param u: The element to find.
        :return: The representative of the set containing u.
        """
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        """
        Unites the sets containing u and v.
        Uses union by rank to keep the tree shallow.
        :param u: First element.
        :param v: Second element.
        """
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# Fisher-Yates Shuffle Algorithm
# Also known as the Knuth Shuffle
def fisher_yates_shuffle(arr):
    """
    Randomly shuffles an array in place using the Fisher-Yates algorithm.
    :param arr: List of elements to shuffle.
    :return: None (shuffles the array in place).
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

# Main execution
if __name__ == "__main__":
    # Test Union-Find (Disjoint Set)
    print("Testing Union-Find (Disjoint Set)")
    n = 5
    ds = DisjointSet(n)
    ds.union(0, 2)
    ds.union(4, 2)
    ds.union(3, 1)
    print(f"Parent of 4: {ds.find(4)}")  # Should be the same as the representative of 0 or 2
    print(f"Parent of 3: {ds.find(3)}")  # Should be the same as the representative of 1
    print()

    # Test Fisher-Yates Shuffle
    print("Testing Fisher-Yates Shuffle")
    array = [1, 2, 3, 4, 5]
    print("Original array:", array)
    fisher_yates_shuffle(array)
    print("Shuffled array:", array)
    print()

