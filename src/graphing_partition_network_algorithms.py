#!/usr/bin/env python3

"""
Module Name: graph_partition_networking.py
Description: Examples for Graph Partitioning and Network Flow Algorithms (Bipartite Graph Check, Hopcroft-Karp Algorithm)
Author: Marvin Billings
Date: 12/02/2024
"""

from collections import deque

# Bipartite Graph Check using BFS
def is_bipartite(graph, start):
    """
    Checks if a graph is bipartite using BFS.
    :param graph: Adjacency list representation of the graph.
    :param start: The starting node.
    :return: True if the graph is bipartite, False otherwise.
    """
    color = {}
    queue = deque([start])
    color[start] = 0  # Assign the first color
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in color:
                # Assign alternate color to the neighbor
                color[neighbor] = 1 - color[node]
                queue.append(neighbor)
            elif color[neighbor] == color[node]:
                # If the neighbor has the same color, the graph is not bipartite
                return False
    return True

# Hopcroft-Karp Algorithm for Maximum Bipartite Matching
class HopcroftKarp:
    def __init__(self, graph):
        self.graph = graph
        self.pair_u = {}  # Pairing for set U
        self.pair_v = {}  # Pairing for set V
        self.dist = {}
        self.U = set(graph.keys())  # Set U is represented by the graph keys
        self.V = set()  # Set V will be gathered from neighbors
        for neighbors in graph.values():
            self.V.update(neighbors)

    def bfs(self):
        """Performs BFS to find augmenting paths."""
        queue = deque()
        for u in self.U:
            if u not in self.pair_u:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = float('inf')
        self.dist[None] = float('inf')
        while queue:
            u = queue.popleft()
            if self.dist[u] < self.dist[None]:
                for v in self.graph[u]:
                    if self.pair_v.get(v) not in self.dist:
                        self.dist[self.pair_v.get(v)] = self.dist[u] + 1
                        queue.append(self.pair_v.get(v))
        return self.dist[None] != float('inf')

    def dfs(self, u):
        """Performs DFS to build matching using augmenting paths."""
        if u is not None:
            for v in self.graph[u]:
                if self.dist[self.pair_v.get(v)] == self.dist[u] + 1:
                    if self.dfs(self.pair_v.get(v)):
                        self.pair_v[v] = u
                        self.pair_u[u] = v
                        return True
            self.dist[u] = float('inf')
            return False
        return True

    def maximum_matching(self):
        """Finds maximum bipartite matching using the Hopcroft-Karp algorithm."""
        matching = 0
        while self.bfs():
            for u in self.U:
                if u not in self.pair_u:
                    if self.dfs(u):
                        matching += 1
        return matching

# Main execution
if __name__ == "__main__":
    # Test Bipartite Graph Check
    print("Testing Bipartite Graph Check")
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A', 'D'],
        'D': ['B', 'C']
    }
    start_node = 'A'
    print(f"Is the graph bipartite starting from node {start_node}? {is_bipartite(graph, start_node)}")
    print()

    # Test Hopcroft-Karp Algorithm
    print("Testing Hopcroft-Karp Algorithm for Maximum Bipartite Matching")
    bipartite_graph = {
        'U1': ['V1', 'V2'],
        'U2': ['V1'],
        'U3': ['V2', 'V3'],
        'U4': ['V3']
    }
    hopcroft_karp = HopcroftKarp(bipartite_graph)
    print(f"Maximum bipartite matching: {hopcroft_karp.maximum_matching()}")

