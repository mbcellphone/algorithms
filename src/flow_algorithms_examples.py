#!/usr/bin/env python3

"""
Module Name: flow_algorithms.py
Description: Examples for Flow Algorithms (Ford-Fulkerson Method, Edmonds-Karp Algorithm)
Author: Marvin Billings
Date: 12/02/2024
"""

from collections import deque

# Ford-Fulkerson Method using BFS (Edmonds-Karp Implementation)
class Graph:
    def __init__(self, graph):
        self.graph = graph  # Residual graph
        self.ROW = len(graph)

    def bfs(self, s, t, parent):
        """
        A BFS-based function to find if there is a path from source 's' to sink 't'.
        :param s: Source vertex.
        :param t: Sink vertex.
        :param parent: List to store the path.
        :return: True if a path exists, False otherwise.
        """
        visited = [False] * self.ROW
        queue = deque([s])
        visited[s] = True
        
        while queue:
            u = queue.popleft()
            for ind, val in enumerate(self.graph[u]):
                if not visited[ind] and val > 0:  # If there is capacity left and not yet visited
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        """
        Returns the maximum flow from source to sink in the given graph using Ford-Fulkerson method.
        :param source: Source vertex.
        :param sink: Sink vertex.
        :return: Maximum flow value.
        """
        parent = [-1] * self.ROW
        max_flow = 0

        # Augment the flow while there is a path from source to sink
        while self.bfs(source, sink, parent):
            # Find the maximum flow through the path found.
            path_flow = float('Inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            
            # update residual capacities of the edges and reverse edges along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            
            max_flow += path_flow
        
        return max_flow

# Main execution
if __name__ == "__main__":
    # Test Ford-Fulkerson (Edmonds-Karp) Algorithm
    print("Testing Ford-Fulkerson (Edmonds-Karp) Algorithm")
    graph = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ]
    g = Graph(graph)
    source = 0
    sink = 5
    print(f"The maximum possible flow is: {g.ford_fulkerson(source, sink)}")

