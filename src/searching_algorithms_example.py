#!/usr/bin/env python3

"""
Module Name: searching_algorithms.py
Description: Examples for Searching Algorithms (Linear Search, Binary Search, Depth-First Search, Breadth-First Search)
Author: Marvin Billings
Date: 12/02/2024
"""

# Linear Search Function
def linear_search(arr, target):
    """
    Performs a linear search on an array.
    :param arr: List of numbers.
    :param target: The value to search for.
    :return: Index of the target if found, else -1.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Binary Search Function
def binary_search(arr, target):
    """
    Performs binary search on a sorted array.
    :param arr: Sorted list of numbers.
    :param target: The value to search for.
    :return: Index of the target if found, else -1.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Depth-First Search (DFS) Function
def dfs(graph, node, visited=None):
    """
    Performs Depth-First Search on a graph.
    :param graph: Adjacency list representation of the graph.
    :param node: The starting node.
    :param visited: Set of visited nodes.
    :return: None (prints nodes in DFS order).
    """
    if visited is None:
        visited = set()
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)

# Breadth-First Search (BFS) Function
def bfs(graph, start):
    """
    Performs Breadth-First Search on a graph.
    :param graph: Adjacency list representation of the graph.
    :param start: The starting node.
    :return: None (prints nodes in BFS order).
    """
    from collections import deque
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)

# Main execution
if __name__ == "__main__":
    # Test Linear Search
    print("Testing Linear Search")
    array = [10, 20, 30, 40, 50]
    target = 30
    print("Array:", array)
    print(f"Target {target} found at index:", linear_search(array, target))
    print()

    # Test Binary Search
    print("Testing Binary Search")
    array = [1, 3, 5, 7, 9, 11, 13]
    target = 7
    print("Array:", array)
    print(f"Target {target} found at index:", binary_search(array, target))
    print()

    # Test Depth-First Search (DFS)
    print("Testing Depth-First Search (DFS)")
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    print("DFS starting from node A:")
    dfs(graph, 'A')
    print("\n")

    # Test Breadth-First Search (BFS)
    print("Testing Breadth-First Search (BFS)")
    print("BFS starting from node A:")
    bfs(graph, 'A')
    print()

