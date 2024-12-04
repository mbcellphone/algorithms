#!/usr/bin/env python3

"""
Module Name: graph_algorithms.py
Description: Examples for Graph Algorithms (Dijkstra's Algorithm, Kruskal's Algorithm, Prim's Algorithm)
Author: Marvin Billings
Date: 12/02/2024
"""

import heapq

# Dijkstra's Algorithm using a Priority Queue
def dijkstra(graph, start):
    """
    Implements Dijkstra's algorithm to find the shortest paths from the start node to all other nodes.
    :param graph: Adjacency list representation of the graph.
    :param start: The starting node.
    :return: Dictionary of shortest distances from start to all nodes.
    """
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Kruskal's Algorithm to find Minimum Spanning Tree (MST)
class DisjointSet:
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, u, v):
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

# Kruskal's MST function
def kruskal(graph):
    """
    Implements Kruskal's algorithm to find the Minimum Spanning Tree (MST).
    :param graph: List of edges represented as (weight, node1, node2).
    :return: List of edges in the MST.
    """
    edges = sorted(graph, key=lambda x: x[0])
    vertices = set()
    for edge in edges:
        vertices.update([edge[1], edge[2]])
    
    disjoint_set = DisjointSet(vertices)
    mst = []
    
    for weight, u, v in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, weight))
    
    return mst

# Prim's Algorithm to find Minimum Spanning Tree (MST)
def prim(graph, start):
    """
    Implements Prim's algorithm to find the Minimum Spanning Tree (MST).
    :param graph: Adjacency list representation of the graph.
    :param start: The starting node.
    :return: List of edges in the MST.
    """
    mst = []
    visited = set([start])
    edges = [(weight, start, to) for to, weight in graph[start].items()]
    heapq.heapify(edges)
    
    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))
            for to_next, weight in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (weight, to, to_next))
    
    return mst

# Main execution
if __name__ == "__main__":
    # Test Dijkstra's Algorithm
    print("Testing Dijkstra's Algorithm")
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    start_node = 'A'
    print(f"Shortest distances from node {start_node}: {dijkstra(graph, start_node)}")
    print()

    # Test Kruskal's Algorithm
    print("Testing Kruskal's Algorithm")
    graph_edges = [
        (1, 'A', 'B'),
        (4, 'A', 'C'),
        (2, 'B', 'C'),
        (5, 'B', 'D'),
        (1, 'C', 'D')
    ]
    print("Edges in the MST:", kruskal(graph_edges))
    print()

    # Test Prim's Algorithm
    print("Testing Prim's Algorithm")
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    start_node = 'A'
    print("Edges in the MST:", prim(graph, start_node))

