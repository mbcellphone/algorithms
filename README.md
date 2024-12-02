# **Introduction to Algorithms**

Welcome to **Introduction to Algorithms**! This repository provides examples of various algorithms from the renowned textbook *"Introduction to Algorithms"* by Cormen, Leiserson, Rivest, and Stein (CLRS). Below is an overview of the different algorithms and the chapters where they can be found.

---

## **Table of Contents**

1. [Algorithm Overview](#algorithm-overview)
2. [Sorting Algorithms](#sorting-algorithms)
   - [Insertion Sort](#insertion-sort-chapter-2-getting-started)
   - [Merge Sort](#merge-sort-chapter-2-divide-and-conquer)
   - [Quicksort](#quicksort-chapter-7-quicksort)
3. [Searching Algorithms](#searching-algorithms)
   - [Binary Search](#binary-search)
   - [Breadth-First Search (BFS)](#breadth-first-search-bfs)
   - [Depth-First Search (DFS)](#depth-first-search-dfs)
4. [Graph Algorithms](#graph-algorithms)
   - [Dijkstra's Algorithm](#dijkstras-algorithm)
   - [Kruskal's Algorithm](#kruskals-algorithm)
   - [Prim's Algorithm](#prims-algorithm)
5. [Dynamic Programming Algorithms](#dynamic-programming-algorithms)
   - [Fibonacci Sequence](#fibonacci-sequence)
   - [Longest Common Subsequence (LCS)](#longest-common-subsequence-lcs)
   - [Knapsack Problem](#knapsack-problem)
6. [Greedy Algorithms](#greedy-algorithms)
   - [Activity Selection Problem](#activity-selection-problem)
   - [Huffman Coding](#huffman-coding)
7. [Divide-and-Conquer Algorithms](#divide-and-conquer-algorithms)
   - [Strassen's Algorithm](#strassens-algorithm)
8. [Backtracking Algorithms](#backtracking-algorithms)
   - [N-Queens Problem](#n-queens-problem)
   - [Sudoku Solver](#sudoku-solver)
9. [Mathematical Algorithms](#mathematical-algorithms)
   - [GCD (Euclidean Algorithm)](#gcd-euclidean-algorithm)
   - [Sieve of Eratosthenes](#sieve-of-eratosthenes)
10. [String Algorithms](#string-algorithms)
    - [Rabin-Karp Algorithm](#rabin-karp-algorithm)
    - [Knuth-Morris-Pratt (KMP) Algorithm](#knuth-morris-pratt-kmp-algorithm)
11. [Tree Algorithms](#tree-algorithms)
    - [Binary Search Tree (BST) Operations](#binary-search-tree-bst-operations)
    - [AVL Trees](#avl-trees)
    - [Red-Black Trees](#red-black-trees)
12. [Flow Algorithms](#flow-algorithms)
    - [Ford-Fulkerson Algorithm](#ford-fulkerson-algorithm)
13. [Miscellaneous Algorithms](#miscellaneous-algorithms)
    - [Union-Find](#union-find)
14. [How to Use](#how-to-use)
15. [Contributing](#contributing)
16. [License](#license)

---

## **Algorithm Overview**

| **Algorithm Type**             | **Algorithms**                              | **CLRS Chapter**                |
|--------------------------------|--------------------------------------------|---------------------------------|
| Sorting Algorithms             | Insertion Sort, Merge Sort, Quicksort      | Chapter 2, 6, 7, 8              |
| Searching Algorithms           | Binary Search, BFS, DFS                    | Chapter 2, 22                   |
| Graph Algorithms               | BFS, DFS, Dijkstra, Kruskal, Prim          | Chapter 22, 23, 24, 25          |
| Dynamic Programming            | Fibonacci, LCS, Knapsack                   | Chapter 15                      |
| Greedy Algorithms              | Activity Selection, Huffman, Kruskal       | Chapter 16, 23                  |
| Divide-and-Conquer Algorithms  | Merge Sort, Quicksort, Strassen's          | Chapter 2, 4, 7                 |
| Backtracking Algorithms        | N-Queens, Sudoku Solver                    | Covered within relevant chapters|
| Mathematical Algorithms        | GCD, Sieve of Eratosthenes                 | Chapter 31                      |
| String Algorithms              | Rabin-Karp, KMP                            | Chapter 32                      |
| Tree Algorithms                | BST, AVL Trees, Red-Black Trees            | Chapter 12, 13                  |
| Flow Algorithms                | Ford-Fulkerson                             | Chapter 26                      |
| Miscellaneous Algorithms       | Union-Find                                 | Chapter 21                      |

---

## **Sorting Algorithms**

### **Insertion Sort (Chapter 2: Getting Started)**
Insertion Sort is a simple algorithm that sorts an array one item at a time by shifting elements to their correct positions.

**Steps**:
1. Iterate through each element of the array.
2. Compare the current element with elements in the sorted portion.
3. Shift elements to make space for the current element in its correct position.

**Python Implementation**:
```python
def insertion_sort(arr):
    """
    Sorts an array using the Insertion Sort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```

**Example Usage**:
```python
if __name__ == "__main__":
    array = [12, 11, 13, 5, 6]
    print("Original array:", array)
    insertion_sort(array)
    print("Sorted array:", array)
```

**Example Output**:
```
Original array: [12, 11, 13, 5, 6]
Sorted array: [5, 6, 11, 12, 13]
```

---

## **Merge Sort (Chapter 2: Divide-and-Conquer)**
Merge Sort is a classic divide-and-conquer algorithm that:
- **Divides** the array into two halves.
- **Recursively** sorts each half.
- **Merges** the sorted halves.

**Python Implementation**:
```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        merge_sort(left_half)
        merge_sort(right_half)
        
        i = j = k = 0
        
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
```

**Example Usage**:
```python
if __name__ == "__main__":
    array = [38, 27, 43, 3, 9, 82, 10]
    print("Original array:", array)
    merge_sort(array)
    print("Sorted array:", array)
```

---

## **Quicksort (Chapter 7: Quicksort)**
Quicksort is another divide-and-conquer algorithm that:
1. **Selects** a pivot element.
2. **Partitions** the array into elements less than and greater than the pivot.
3. **Recursively** sorts the left and right subarrays.

**Python Implementation**:
```python
def quicksort(arr):
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_helper(low, high):
        if low < high:
            pivot_index = partition(low, high)
            quicksort_helper(low, pivot_index - 1)
            quicksort_helper(pivot_index + 1, high)

    quicksort_helper(0, len(arr) - 1)
```

**Example Usage**:
```python
if __name__ == "__main__":
    array = [10, 7, 8, 9, 1, 5]
    print("Original array:", array)
    quicksort(array)
    print("Sorted array:", array)
```

**Example Output**:
```
Original array: [10, 7, 8, 9, 1, 5]
Sorted array: [1, 5, 7, 8, 9, 10]
```
## **Searching Algorithms**
---

---

### **Breadth-First Search (BFS)**
### **Breadth-First Search (BFS) (Chapter 22: Elementary Graph Algorithms)**
Breadth-First Search (BFS) is an algorithm for traversing or searching tree or graph data structures. It explores all nodes at the present depth level before moving on to nodes at the next depth level.

**Python Implementation**:
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Example Usage**:
```python
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    bfs(graph, 'A')
```

**Example Output**:
```
A B C D E F
```

---
### **Depth-First Search (DFS)**
### **Depth-First Search (DFS) (Chapter 22: Elementary Graph Algorithms)**
Depth-First Search (DFS) is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node and explores as far as possible along each branch before backtracking.

**Python Implementation**:
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

**Example Usage**:
```python
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    dfs(graph, 'A')
```

**Example Output**:
```
A B D E F C
```
## **Graph Algorithms**

---
### **Dijkstra's Algorithm**

Certainly! Here is the **Dijkstra's Algorithm** section:

---

### **Dijkstra's Algorithm (Chapter 24: Single-Source Shortest Paths)**
Dijkstra's Algorithm is used to find the shortest paths between nodes in a graph, which may represent, for example, road networks. It is particularly effective for graphs with non-negative edge weights.

**Steps**:
1. Initialize the distance to the start node as 0 and to all other nodes as infinity.
2. Set the starting node as current and mark all other nodes as unvisited.
3. For the current node, consider all its unvisited neighbors and calculate their tentative distances.
4. Once all neighbors of the current node have been considered, mark the current node as visited. A visited node will not be checked again.
5. Select the unvisited node that is marked with the smallest tentative distance and set it as the new current node. Repeat until all nodes are visited or the smallest tentative distance is infinity.

**Python Implementation**:
```python
import heapq

def dijkstra(graph, start):
    """
    Finds the shortest paths from the start node to all other nodes in the graph.
    :param graph: A dictionary representing the adjacency list of the graph. The value is a list of tuples (neighbor, weight).
    :param start: The starting node.
    :return: A dictionary containing the shortest distance to each node.
    """
    pq = []
    heapq.heappush(pq, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```

**Example Usage**:
```python
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    start_node = 'A'
    distances = dijkstra(graph, start_node)
    print(f"Shortest distances from {start_node}: {distances}")
```

**Example Output**:
```
Shortest distances from A: {'A': 0, 'B': 1, 'C': 3, 'D': 4}
---

### **Kruskal's Algorithm**

Certainly! Here is the **Kruskal's Algorithm** section:

---

### **Kruskal's Algorithm (Chapter 23: Minimum Spanning Trees)**
Kruskal's Algorithm is a greedy algorithm that finds a Minimum Spanning Tree (MST) for a connected, weighted graph. It finds the subset of edges that form a tree including all the vertices, where the total weight of all the edges is minimized.

**Steps**:
1. Sort all edges in non-decreasing order of their weights.
2. Pick the smallest edge. Check if it forms a cycle with the MST formed so far.
   - If no cycle is formed, add this edge to the MST.
   - If a cycle is formed, discard the edge.
3. Repeat step 2 until there are \( V - 1 \) edges in the MST, where \( V \) is the number of vertices.

**Python Implementation**:
```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal_mst(self):
        result = []
        i, e = 0, 0

        # Step 1: Sort all the edges in non-decreasing order of their weight
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        # Print the resulting MST
        print("Constructed MST:")
        for u, v, weight in result:
            print(f"{u} -- {v} == {weight}")
```

**Example Usage**:
```python
if __name__ == "__main__":
    g = Graph(4)
    g.add_edge(0, 1, 10)
    g.add_edge(0, 2, 6)
    g.add_edge(0, 3, 5)
    g.add_edge(1, 3, 15)
    g.add_edge(2, 3, 4)

    g.kruskal_mst()
```

**Example Output**:
```
Constructed MST:
2 -- 3 == 4
0 -- 3 == 5
0 -- 1 == 10
```
### **Prim's Algorithm**

---

### **Prim's Algorithm (Chapter 23: Minimum Spanning Trees)**
Prim's Algorithm is a greedy algorithm that finds a Minimum Spanning Tree (MST) for a connected, weighted graph. The algorithm maintains two sets of vertices:
- One set contains the vertices included in the MST.
- The other set contains the vertices not yet included.

Prim's Algorithm starts with an arbitrary vertex and continues until all the vertices are included in the MST.

**Steps**:
1. Start with any vertex and add it to the MST.
2. At each step, add the smallest edge connecting a vertex in the MST to a vertex outside the MST.
3. Repeat until all vertices are included in the MST.

**Python Implementation**:
```python
import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def print_mst(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i} \t{self.graph[i][parent[i]]}")

    def min_key(self, key, mst_set):
        min_value = sys.maxsize
        min_index = -1

        for v in range(self.V):
            if key[v] < min_value and not mst_set[v]:
                min_value = key[v]
                min_index = v

        return min_index

    def prim_mst(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1

        for _ in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True

            for v in range(self.V):
                if self.graph[u][v] > 0 and not mst_set[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.print_mst(parent)
```

**Example Usage**:
```python
if __name__ == "__main__":
    g = Graph(5)
    g.graph = [
        [0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0]
    ]

    g.prim_mst()
```

**Example Output**:
```
Edge    Weight
0 - 1   2
1 - 2   3
0 - 3   6
1 - 4   5
---

---
---


## **Dynamic Programming Algorithms**

### **Fibonacci Sequence**
### **Fibonacci Sequence (Chapter 15: Dynamic Programming)**
The Fibonacci Sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. Dynamic programming can be used to solve this problem efficiently by storing previous results to avoid redundant calculations.

**Python Implementation**:
```python
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib[n]
```

**Example Usage**:
```python
if __name__ == "__main__":
    n = 9
    print(f"Fibonacci number at index {n} is {fibonacci(n)}")
```

**Example Output**:
```
Fibonacci number at index 9 is 34
```

---
## **Longest Common Subsequence (LCS)**
### **Longest Common Subsequence (LCS) (Chapter 15: Dynamic Programming)**
The Longest Common Subsequence (LCS) is the longest sequence that can be derived from two sequences by deleting some elements without changing the order of the remaining elements.

**Python Implementation**:
```python
def lcs(X, Y):
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
```

**Example Usage**:
```python
if __name__ == "__main__":
    X = "AGGTAB"
    Y = "GXTXAYB"
    print(f"Length of LCS is {lcs(X, Y)}")
```

**Example Output**:
```
Length of LCS is 4
```

---
## **Knapsack Problem**
### **Knapsack Problem (Chapter 15: Dynamic Programming)**
The Knapsack Problem is a problem in combinatorial optimization: given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit, and the total value is as large as possible.

**Python Implementation**:
```python
def knapsack(W, wt, val, n):
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
```

**Example Usage**:
```python
if __name__ == "__main__":
    val = [60, 100, 120]
    wt = [10, 20, 30]
    W = 50
    n = len(val)
    print(f"Maximum value in knapsack is {knapsack(W, wt, val, n)}")
```

**Example Output**:
```
Maximum value in knapsack is 220

---

---

---

## **Greedy Algorithms**

---
## **Activity Selection Problem**
---


### **Activity Selection Problem (Chapter 16: Greedy Algorithms)**
The Activity Selection Problem is a problem of selecting the maximum number of activities that don't overlap, given their start and end times.

**Steps**:
1. Sort activities by their ending times.
2. Select the first activity and add it to the result.
3. For each subsequent activity, if the start time is greater than or equal to the end time of the previously selected activity, select it.

**Python Implementation**:
```python
def activity_selection(start, end):
    n = len(start)
    selected_activities = [0]
    
    # The last activity selected
    last_end_time = end[0]
    
    for i in range(1, n):
        if start[i] >= last_end_time:
            selected_activities.append(i)
            last_end_time = end[i]

    return selected_activities
```

**Example Usage**:
```python
if __name__ == "__main__":
    start = [1, 3, 0, 5, 8, 5]
    end = [2, 4, 6, 7, 9, 9]
    selected = activity_selection(start, end)
    print(f"Selected activities: {selected}")
```

**Example Output**:
```
Selected activities: [0, 1, 3, 4]
```

---
## **Huffman Coding**

### **Huffman Coding (Chapter 16: Greedy Algorithms)**
Huffman Coding is a greedy algorithm used for data compression. It builds a variable-length prefix code for each character, where frequently occurring characters have shorter codes. 

**Steps**:
1. Create a leaf node for each character and add it to the priority queue.
2. While there is more than one node in the queue:
   - Remove the two nodes of the highest priority (lowest frequency).
   - Create a new internal node with these two nodes as children and with a frequency equal to the sum of the two nodes' frequencies.
   - Add the new node to the priority queue.
3. The remaining node is the root of the Huffman Tree, and the paths to the leaf nodes represent the Huffman codes.

**Python Implementation**:
```python
import heapq
from collections import Counter, namedtuple

class Node(namedtuple("Node", ["char", "freq"])):
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_tree():
    data = "this is an example for huffman encoding"
    char_freq = Counter(data)
    root = huffman_coding(char_freq)
    return root
```

**Example Usage**:
```python
if __name__ == "__main__":
    root = build_huffman_tree()
    print("Huffman Tree built successfully.")
```

**Example Output**:
```
Huffman Tree built successfully.
```

---




## **Divide-and-Conquer Algorithms**
---
## **Strassen's Algorithm**
Strassen’s Algorithm (Chapter 4: Divide-and-Conquer)

Strassen’s Algorithm is an efficient algorithm for matrix multiplication that uses a divide-and-conquer approach. It improves upon the traditional ￼ complexity of matrix multiplication by reducing the number of recursive multiplications needed.

Steps:
	1.	Divide two ￼ matrices into four submatrices of size ￼.
	2.	Use Strassen’s formulas to compute intermediate products using only 7 multiplications (as opposed to 8 in the naive divide-and-conquer).
	3.	Combine the intermediate results to get the final product matrix.

Mathematical Breakdown:
For two matrices ￼ and ￼:

￼

Split ￼ and ￼ into four submatrices:

￼

Strassen’s algorithm uses the following intermediate products:
	1.	￼
	2.	￼
	3.	￼
	4.	￼
	5.	￼
	6.	￼
	7.	￼

The resulting submatrices of ￼ are:

￼
￼
￼
￼

Python Implementation:

import numpy as np

def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]

def subtract_matrices(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]

def strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    M1 = strassen(add_matrices(A11, A22), add_matrices(B11, B22))
    M2 = strassen(add_matrices(A21, A22), B11)
    M3 = strassen(A11, subtract_matrices(B12, B22))
    M4 = strassen(A22, subtract_matrices(B21, B11))
    M5 = strassen(add_matrices(A11, A12), B22)
    M6 = strassen(subtract_matrices(A21, A11), add_matrices(B11, B12))
    M7 = strassen(subtract_matrices(A12, A22), add_matrices(B21, B22))

    C11 = add_matrices(subtract_matrices(add_matrices(M1, M4), M5), M7)
    C12 = add_matrices(M3, M5)
    C21 = add_matrices(M2, M4)
    C22 = add_matrices(subtract_matrices(add_matrices(M1, M3), M2), M6)

    C = [[0] * n for _ in range(n)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]

    return C

# Example usage
if __name__ == "__main__":
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]

    result = strassen(A, B)
    print("Result of Strassen's Matrix Multiplication:")
    for row in result:
        print(row)

Example Output:

Result of Strassen's Matrix Multiplication:
[80, 70, 60, 50]
[240, 214, 188, 162]
[400, 358, 316, 274]
[560, 502, 444, 386]

Strassen’s algorithm is more efficient than the naive matrix multiplication algorithm for large matrices, as it reduces the time complexity from ￼ to approximately ￼. However, it comes at the cost of additional complexity in terms of implementation and increased space usage for small matrices, which can make the naive algorithm preferable in some scenarios.



---
## **Backtracking Algorithms**

---
## **N-Queens Problem**
Certainly! Here is the Backtracking Algorithms section:

Backtracking Algorithms

Backtracking is a general algorithmic technique for solving problems incrementally, one piece at a time, and removing solutions that fail to satisfy the problem’s constraints at any point. It is particularly useful for constraint satisfaction problems, such as puzzles or combinatorial optimization problems.

N-Queens Problem (Chapter 8: Backtracking Techniques)

The N-Queens problem is the classic example of a backtracking algorithm, where you need to place N queens on an N x N chessboard such that no two queens attack each other.

Steps:
	1.	Start placing queens one by one in different columns.
	2.	If placing a queen does not lead to a solution, backtrack and place the queen in the next possible position.
	3.	Continue until all queens are placed or all possibilities are exhausted.

Python Implementation:

def print_solution(board):
    for row in board:
        print(" ".join(str(x) for x in row))
    print()

def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 1:
            return False

    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, col, n):
    if col >= n:
        print_solution(board)
        return True

    res = False
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1
            res = solve_n_queens_util(board, col + 1, n) or res
            board[i][col] = 0

    return res

def solve_n_queens(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    if not solve_n_queens_util(board, 0, n):
        print("No solution exists")

Example Usage:

if __name__ == "__main__":
    n = 4
    solve_n_queens(n)

Example Output:

0 0 1 0
1 0 0 0
0 0 0 1
0 1 0 0

0 1 0 0
0 0 0 1
1 0 0 0
0 0 1 0

---
## **Sudoku Solver**
Sudoku Solver (Chapter 8: Backtracking Techniques)

The Sudoku Solver is another example of a backtracking algorithm. The goal is to fill in the empty cells in a 9x9 Sudoku grid such that each number from 1 to 9 appears exactly once in each row, column, and 3x3 subgrid.

Steps:
	1.	Find an empty cell in the grid.
	2.	Attempt to place a number from 1 to 9 in the cell.
	3.	Check if placing the number is valid.
	4.	If it is valid, recursively attempt to solve the rest of the board.
	5.	If placing a number leads to an invalid configuration, backtrack and try the next number.

Python Implementation:

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def print_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

Example Usage:

if __name__ == "__main__":
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    if solve_sudoku(board):
        print_board(board)
    else:
        print("No solution exists")

Example Output:

5 3 4 6 7 8 9 1 2
6 7 2 1 9 5 3 4 8
1 9 8 3 4 2 5 6 7
8 5 9 7 6 1 4 2 3
4 2 6 8 5 3 7 9 1
7 1 3 9 2 4 8 5 6
9 6 1 5 3 7 2 8 4
2 8 7 4 1 9 6 3 5
3 4 5 2 8 6 1 7 9

Let me know if you need any more details or additional examples on Backtracking Algorithms!


---
## **Mathematical Algorithms**
---
## **GCD (Euclidean Algorithm)**
Certainly! Here is the GCD (Greatest Common Divisor) Algorithm section:

GCD (Euclidean Algorithm) (Chapter 31: Number-Theoretic Algorithms)

The Greatest Common Divisor (GCD) of two integers is the largest positive integer that divides both numbers without leaving a remainder. The Euclidean Algorithm is an efficient way to calculate the GCD, based on the principle that the GCD of two numbers does not change if the larger number is replaced by its remainder when divided by the smaller number.

Steps:
	1.	Start with two numbers, ￼ and ￼.
	2.	Replace ￼ with ￼ and ￼ with the remainder of ￼ divided by ￼.
	3.	Repeat until ￼ becomes zero. The value of ￼ at this point is the GCD.

Python Implementation:

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

Example Usage:

if __name__ == "__main__":
    a = 56
    b = 98
    result = gcd(a, b)
    print(f"The GCD of {a} and {b} is {result}")

Example Output:

The GCD of 56 and 98 is 14

Explanation:
	•	Start with ￼, ￼.
	•	Compute ￼ (i.e., ￼), which gives 56 (since ￼).
	•	Swap values: ￼, ￼.
	•	Compute ￼ (i.e., ￼), which gives 42.
	•	Swap values: ￼, ￼.
	•	Continue this process until ￼, at which point ￼ is the GCD.

Recursive Implementation:

The Euclidean algorithm can also be implemented recursively.

Python Implementation:

def gcd_recursive(a, b):
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)

Example Usage:

if __name__ == "__main__":
    a = 56
    b = 98
    result = gcd_recursive(a, b)
    print(f"The GCD of {a} and {b} using the recursive method is {result}")

Example Output:

The GCD of 56 and 98 using the recursive method is 14

Use Cases:
	•	Simplifying Fractions: GCD is used to simplify fractions by dividing both the numerator and the denominator by their GCD.
	•	Cryptography: GCD calculations are often used in cryptographic algorithms such as RSA.

The Euclidean Algorithm is very efficient with a time complexity of ￼, making it suitable for large numbers.

Let me know if you need any more details or examples regarding the GCD algorithm or any other topics!
---
## **Sieve of Eratosthenes**
Certainly! Here is the Sieve of Eratosthenes section:

Sieve of Eratosthenes (Chapter 31: Number-Theoretic Algorithms)

The Sieve of Eratosthenes is an efficient algorithm for finding all prime numbers up to a specified integer ￼. It systematically marks the multiples of each prime number starting from 2, effectively leaving only the prime numbers unmarked.

Steps:
	1.	Create an array is_prime of size ￼ and initialize all entries as True. Set the values for 0 and 1 as False since 0 and 1 are not prime numbers.
	2.	Starting with the first prime number (2), mark all of its multiples as False.
	3.	Move to the next unmarked number and repeat the process until you’ve processed numbers up to ￼.
	4.	The remaining True entries in the array represent prime numbers.

Python Implementation:

def sieve_of_eratosthenes(n):
    # Initialize an array of booleans representing if numbers are prime
    is_prime = [True] * (n + 1)
    is_prime[0], is_prime[1] = False, False  # 0 and 1 are not prime

    p = 2
    while p * p <= n:
        if is_prime[p]:
            # Mark all multiples of p as False
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1

    # Collect all prime numbers from the is_prime array
    primes = [num for num, prime in enumerate(is_prime) if prime]
    return primes

Example Usage:

if __name__ == "__main__":
    n = 30
    primes = sieve_of_eratosthenes(n)
    print(f"Prime numbers up to {n}: {primes}")

Example Output:

Prime numbers up to 30: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

Explanation:
	•	The algorithm starts by marking all multiples of 2 (except 2 itself) as non-prime.
	•	It proceeds to the next number, 3, and marks all multiples of 3 as non-prime.
	•	This process continues for the next unmarked number (which is 5, then 7, and so on) until ￼.

Complexity:
	•	Time Complexity: ￼
	•	Space Complexity: ￼

The Sieve of Eratosthenes is an efficient way to find all prime numbers less than or equal to a given limit, making it particularly useful for problems involving prime numbers.

Optimizations:
	•	You can reduce memory usage by only storing odd numbers since even numbers (except 2) are not prime.
	•	Instead of starting to mark multiples from ￼, another optimization is to mark multiples starting from ￼ if you have previously marked smaller multiples.



---
## **String Algorithms**
---
## **Rabin-Karp Algorithm**
---Certainly! Here is the Rabin-Karp Algorithm section:

Rabin-Karp Algorithm (Chapter 32: String Matching Algorithms)

The Rabin-Karp Algorithm is a string-searching algorithm used for searching a substring (pattern) within a main string (text). It uses hashing to find an exact match between the pattern and any substring of the text, allowing for efficient multiple pattern matching. It is particularly useful when you need to search for multiple patterns simultaneously.

Steps:
	1.	Calculate the hash value of the pattern and the initial substring of the text of the same length.
	2.	Slide the pattern over the text one character at a time.
	3.	For each substring, calculate its hash value and compare it with the pattern’s hash value.
	•	If the hash values are equal, compare the actual characters to confirm a match (to avoid hash collisions).
	•	If the hash values are not equal, continue sliding.
	4.	Recalculate the hash for the next substring in constant time using a rolling hash technique.

Hash Function:
The hash function used is typically a polynomial rolling hash function that can be computed efficiently and has a low probability of collisions.

Python Implementation:

def rabin_karp(text, pattern, q=101):
    d = 256  # Number of characters in the input alphabet
    m = len(pattern)
    n = len(text)
    h = 1
    p = 0  # Hash value for the pattern
    t = 0  # Hash value for the text
    results = []

    # The value of h would be "pow(d, m-1) % q"
    for i in range(m - 1):
        h = (h * d) % q

    # Calculate the hash value of the pattern and the first window of text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over the text
    for i in range(n - m + 1):
        # Check the hash values of the current window of text and the pattern
        if p == t:
            # If the hash values match, check characters one by one
            if text[i:i + m] == pattern:
                results.append(i)

        # Calculate hash value for the next window of text
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            # Convert negative hash value to positive
            if t < 0:
                t = t + q

    return results

Example Usage:

if __name__ == "__main__":
    text = "ABCCDABCDABAB"
    pattern = "AB"
    matches = rabin_karp(text, pattern)
    print(f"Pattern found at positions: {matches}")

Example Output:

Pattern found at positions: [0, 7, 11]

Explanation:
	•	Initial Hash Calculation: Compute the hash values for the pattern and the first window of the text.
	•	Sliding Window: Use a sliding window of length equal to the pattern and compute the hash value for each substring.
	•	Collision Checking: If the hash values of the pattern and the substring match, confirm by comparing the actual characters to rule out false positives.
	•	Rolling Hash Update: Recalculate the hash of the next substring by removing the influence of the first character and adding the influence of the new character, which makes it efficient.

Advantages:
	•	Efficient for multiple pattern searches.
	•	The use of a rolling hash function makes recalculating hash values very efficient, requiring constant time on average.

Time Complexity:
	•	Average Case: ￼ where ￼ is the length of the text and ￼ is the length of the pattern.
	•	Worst Case: ￼, in case of hash collisions for each substring.

Use Cases:
	•	Plagiarism Detection: Finding occurrences of sentences or paragraphs in large documents.
	•	Pattern Matching: Searching for specific DNA sequences in biological databases.

Let me know if you need further details, examples, or any other algorithms!

---
## **Knuth-Morris-Pratt (KMP) Algorithm**
Certainly! Here is the Knuth-Morris-Pratt (KMP) Algorithm section:

Knuth-Morris-Pratt (KMP) Algorithm (Chapter 32: String Matching Algorithms)

The Knuth-Morris-Pratt (KMP) Algorithm is an efficient pattern-searching algorithm that finds the occurrences of a “pattern” within a “text” by preprocessing the pattern to determine how much of it can be reused. It uses an auxiliary array (LPS - Longest Prefix Suffix) to avoid unnecessary comparisons, thus reducing the time complexity of the search.

Steps:
	1.	Preprocess the Pattern: Create the LPS (Longest Prefix Suffix) array, which represents the longest proper prefix that is also a suffix for each sub-pattern ending at different positions in the pattern.
	2.	Search the Pattern: Use the LPS array to skip characters that have already been matched, minimizing redundant comparisons.

LPS Array:
The LPS array is used to skip unnecessary comparisons by storing the lengths of the longest proper prefixes which are also suffixes for each position of the pattern. It helps to decide where the next match should start without rechecking already matched characters.

Python Implementation:

Step 1: Compute LPS Array:

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0  # Length of the previous longest prefix suffix
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

Step 2: KMP Search:

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    lps = compute_lps(pattern)
    
    i = 0  # Index for text
    j = 0  # Index for pattern
    results = []

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            results.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results

Example Usage:

if __name__ == "__main__":
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    matches = kmp_search(text, pattern)
    print(f"Pattern found at positions: {matches}")

Example Output:

Pattern found at positions: [10]

Explanation:
	1.	LPS Array Calculation:
	•	For the pattern "ABABCABAB", the LPS array helps to know how much we can skip in case of a mismatch.
	•	The LPS array for the given pattern will be [0, 0, 1, 2, 0, 1, 2, 3, 4].
	2.	Pattern Searching:
	•	The KMP algorithm traverses through the text, using the LPS array to skip redundant comparisons.
	•	When a mismatch occurs after a match, instead of rechecking previously matched characters, the algorithm uses the LPS array to determine the next position of comparison.

Advantages:
	•	Efficient Search: It uses preprocessing to enable faster pattern matching by minimizing backtracking.
	•	Time Complexity:
	•	Preprocessing: ￼, where ￼ is the length of the pattern.
	•	Searching: ￼, where ￼ is the length of the text.
	•	Overall: ￼.

Use Cases:
	•	Text Editors: Searching for words in documents.
	•	Biological Sequence Analysis: Searching specific sequences in DNA strings.

The Knuth-Morris-Pratt Algorithm is highly efficient in terms of reducing the number of character comparisons, making it ideal for applications that require repeated searching of patterns.

Let me know if you need more details or any further examples on KMP or other string matching algorithms!
---
## **Tree Algorithms**
Certainly! Here is the Tree Algorithms section:

Tree Algorithms

Tree algorithms are fundamental data structures used to solve various computational problems efficiently. Below, we cover key tree algorithms, including Binary Search Tree (BST) Operations, AVL Trees, and Red-Black Trees.

Binary Search Tree (BST) Operations (Chapter 12: Binary Search Trees)

A Binary Search Tree (BST) is a binary tree in which each node has a key greater than all keys in its left subtree and smaller than all keys in its right subtree. BSTs allow for efficient searching, insertion, and deletion operations.

Operations:
	1.	Insertion: Insert a node while maintaining the BST property.
	2.	Search: Search for a node by leveraging the ordered nature of the tree.
	3.	Deletion: Remove a node while ensuring the BST property remains intact.

Python Implementation:

BST Node Class:

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if key < root.key:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
    return root

def search(root, key):
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search(root.left, key)
    return search(root.right, key)

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.key, end=" ")
        inorder_traversal(root.right)

Example Usage:

if __name__ == "__main__":
    root = None
    keys = [50, 30, 20, 40, 70, 60, 80]

    for key in keys:
        root = insert(root, key)

    print("Inorder traversal of the BST:")
    inorder_traversal(root)

    key_to_search = 40
    found_node = search(root, key_to_search)
    if found_node:
        print(f"\nKey {key_to_search} found in the BST.")
    else:
        print(f"\nKey {key_to_search} not found in the BST.")

Example Output:

Inorder traversal of the BST:
20 30 40 50 60 70 80 
Key 40 found in the BST.

AVL Trees (Chapter 13: Balanced Trees)

AVL Trees are self-balancing binary search trees where the difference between the heights of the left and right subtrees for every node is no more than one. The balance is maintained by rotating the tree when necessary.

Operations:
	1.	Rotations: Single or double rotations are used to maintain the balance factor of each node.
	2.	Insertion: Insert a node and then rebalance the tree if necessary.
	3.	Deletion: Remove a node and perform rotations to rebalance the tree.

Python Implementation (Insertion Example):

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

def get_height(node):
    if not node:
        return 0
    return node.height

def right_rotate(y):
    x = y.left
    T2 = x.right

    x.right = y
    y.left = T2

    y.height = 1 + max(get_height(y.left), get_height(y.right))
    x.height = 1 + max(get_height(x.left), get_height(x.right))

    return x

def left_rotate(x):
    y = x.right
    T2 = y.left

    y.left = x
    x.right = T2

    x.height = 1 + max(get_height(x.left), get_height(x.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))

    return y

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def insert_avl(node, key):
    if not node:
        return AVLNode(key)
    elif key < node.key:
        node.left = insert_avl(node.left, key)
    else:
        node.right = insert_avl(node.right, key)

    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)

    if balance > 1 and key < node.left.key:
        return right_rotate(node)
    if balance < -1 and key > node.right.key:
        return left_rotate(node)
    if balance > 1 and key > node.left.key:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    if balance < -1 and key < node.right.key:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node

Example Usage:

if __name__ == "__main__":
    root = None
    keys = [10, 20, 30, 40, 50, 25]

    for key in keys:
        root = insert_avl(root, key)

    print("AVL Tree constructed successfully.")

Example Output:

AVL Tree constructed successfully.

Red-Black Trees (Chapter 13: Balanced Trees)

Red-Black Trees are balanced binary search trees that ensure the tree remains approximately balanced, maintaining efficient search times. They enforce properties using color assignments (red or black) to nodes.

Properties:
	1.	Every node is either red or black.
	2.	The root is always black.
	3.	Red nodes cannot have red children (no two consecutive red nodes).
	4.	Every path from a node to its descendant null nodes has the same number of black nodes.

Operations:
	•	Insertion and Recoloring: Nodes are inserted like a normal BST, and then recoloring or rotations are performed to maintain properties.
	•	Rotation Types: Rotations (left or right) are used to correct any imbalance while maintaining the properties.

These tree algorithms, including BSTs, AVL Trees, and Red-Black Trees, are crucial for maintaining efficient data structures that support fast insertion, deletion, and search operations.

Let me know if you need further details or specific examples on any of these tree algorithms!

---
## **Binary Search Tree (BST) Operations**
Certainly! Here is the Tree Algorithms section:

Tree Algorithms

Tree algorithms are fundamental data structures used to solve various computational problems efficiently. Below, we cover key tree algorithms, including Binary Search Tree (BST) Operations, AVL Trees, and Red-Black Trees.

Binary Search Tree (BST) Operations (Chapter 12: Binary Search Trees)

A Binary Search Tree (BST) is a binary tree in which each node has a key greater than all keys in its left subtree and smaller than all keys in its right subtree. BSTs allow for efficient searching, insertion, and deletion operations.

Operations:
	1.	Insertion: Insert a node while maintaining the BST property.
	2.	Search: Search for a node by leveraging the ordered nature of the tree.
	3.	Deletion: Remove a node while ensuring the BST property remains intact.

Python Implementation:

BST Node Class:

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if key < root.key:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
    return root

def search(root, key):
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search(root.left, key)
    return search(root.right, key)

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.key, end=" ")
        inorder_traversal(root.right)

Example Usage:

if __name__ == "__main__":
    root = None
    keys = [50, 30, 20, 40, 70, 60, 80]

    for key in keys:
        root = insert(root, key)

    print("Inorder traversal of the BST:")
    inorder_traversal(root)

    key_to_search = 40
    found_node = search(root, key_to_search)
    if found_node:
        print(f"\nKey {key_to_search} found in the BST.")
    else:
        print(f"\nKey {key_to_search} not found in the BST.")

Example Output:

Inorder traversal of the BST:
20 30 40 50 60 70 80 
Key 40 found in the BST.

AVL Trees (Chapter 13: Balanced Trees)

AVL Trees are self-balancing binary search trees where the difference between the heights of the left and right subtrees for every node is no more than one. The balance is maintained by rotating the tree when necessary.

Operations:
	1.	Rotations: Single or double rotations are used to maintain the balance factor of each node.
	2.	Insertion: Insert a node and then rebalance the tree if necessary.
	3.	Deletion: Remove a node and perform rotations to rebalance the tree.

Python Implementation (Insertion Example):

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

def get_height(node):
    if not node:
        return 0
    return node.height

def right_rotate(y):
    x = y.left
    T2 = x.right

    x.right = y
    y.left = T2

    y.height = 1 + max(get_height(y.left), get_height(y.right))
    x.height = 1 + max(get_height(x.left), get_height(x.right))

    return x

def left_rotate(x):
    y = x.right
    T2 = y.left

    y.left = x
    x.right = T2

    x.height = 1 + max(get_height(x.left), get_height(x.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))

    return y

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def insert_avl(node, key):
    if not node:
        return AVLNode(key)
    elif key < node.key:
        node.left = insert_avl(node.left, key)
    else:
        node.right = insert_avl(node.right, key)

    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)

    if balance > 1 and key < node.left.key:
        return right_rotate(node)
    if balance < -1 and key > node.right.key:
        return left_rotate(node)
    if balance > 1 and key > node.left.key:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    if balance < -1 and key < node.right.key:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node

Example Usage:

if __name__ == "__main__":
    root = None
    keys = [10, 20, 30, 40, 50, 25]

    for key in keys:
        root = insert_avl(root, key)

    print("AVL Tree constructed successfully.")

Example Output:

AVL Tree constructed successfully.

Red-Black Trees (Chapter 13: Balanced Trees)

Red-Black Trees are balanced binary search trees that ensure the tree remains approximately balanced, maintaining efficient search times. They enforce properties using color assignments (red or black) to nodes.

Properties:
	1.	Every node is either red or black.
	2.	The root is always black.
	3.	Red nodes cannot have red children (no two consecutive red nodes).
	4.	Every path from a node to its descendant null nodes has the same number of black nodes.

Operations:
	•	Insertion and Recoloring: Nodes are inserted like a normal BST, and then recoloring or rotations are performed to maintain properties.
	•	Rotation Types: Rotations (left or right) are used to correct any imbalance while maintaining the properties.

These tree algorithms, including BSTs, AVL Trees, and Red-Black Trees, are crucial for maintaining efficient data structures that support fast insertion, deletion, and search operations.


---
## **AVL Trees**
It seems like you may have meant AVL Tree. Here is a more in-depth look at AVL Trees.

AVL Trees (Chapter 13: Balanced Trees)

An AVL Tree (named after its inventors Adelson-Velsky and Landis) is a type of self-balancing binary search tree (BST) where the height of the left and right subtrees of every node differs by at most one. This ensures that the tree remains approximately balanced, which keeps the operations of insertion, deletion, and lookup efficient with a time complexity of ￼.

The primary advantage of an AVL tree is that it ensures a balanced tree structure, preventing the worst-case performance of ordinary binary search trees.

Key Properties:
	1.	Height-Balanced: The difference in height between the left and right subtrees of any node is no more than 1.
	2.	Rotations for Rebalancing: After any insertion or deletion, the tree must be checked for balance. If the balance factor (difference in heights of left and right subtrees) exceeds the allowed limit, rotations are performed to restore balance.

Balance Factor:
	•	The balance factor of a node is defined as the height difference between its left and right subtrees:
￼
	•	A node with a balance factor of -1, 0, or 1 is considered balanced. Any other value requires rebalancing.

Rotations in AVL Trees:
	1.	Right Rotation (LL Rotation): Used when the left subtree is heavier (i.e., balance factor > 1) and the imbalance is in the left subtree of the left child.
	2.	Left Rotation (RR Rotation): Used when the right subtree is heavier (i.e., balance factor < -1) and the imbalance is in the right subtree of the right child.
	3.	Left-Right Rotation (LR Rotation): Used when the left subtree is heavier, but the imbalance is in the right subtree of the left child.
	4.	Right-Left Rotation (RL Rotation): Used when the right subtree is heavier, but the imbalance is in the left subtree of the right child.

Python Implementation:

Below is a Python implementation of the AVL Tree, including insertion, balancing, and rotations:

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

def get_height(node):
    if not node:
        return 0
    return node.height

def right_rotate(y):
    x = y.left
    T2 = x.right

    # Perform rotation
    x.right = y
    y.left = T2

    # Update heights
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    x.height = 1 + max(get_height(x.left), get_height(x.right))

    # Return new root
    return x

def left_rotate(x):
    y = x.right
    T2 = y.left

    # Perform rotation
    y.left = x
    x.right = T2

    # Update heights
    x.height = 1 + max(get_height(x.left), get_height(x.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))

    # Return new root
    return y

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def insert_avl(node, key):
    # Step 1 - Perform normal BST insertion
    if not node:
        return AVLNode(key)
    elif key < node.key:
        node.left = insert_avl(node.left, key)
    else:
        node.right = insert_avl(node.right, key)

    # Step 2 - Update height of this ancestor node
    node.height = 1 + max(get_height(node.left), get_height(node.right))

    # Step 3 - Get the balance factor to check if the node is unbalanced
    balance = get_balance(node)

    # If the node becomes unbalanced, there are 4 cases:

    # Left Left Case
    if balance > 1 and key < node.left.key:
        return right_rotate(node)

    # Right Right Case
    if balance < -1 and key > node.right.key:
        return left_rotate(node)

    # Left Right Case
    if balance > 1 and key > node.left.key:
        node.left = left_rotate(node.left)
        return right_rotate(node)

    # Right Left Case
    if balance < -1 and key < node.right.key:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    # Return the (unchanged) node pointer
    return node

# Utility function to print in-order traversal of the tree
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.key, end=" ")
        inorder_traversal(root.right)

# Example usage
if __name__ == "__main__":
    root = None
    keys = [10, 20, 30, 40, 50, 25]

    for key in keys:
        root = insert_avl(root, key)

    print("Inorder traversal of the constructed AVL tree:")
    inorder_traversal(root)

Example Output:

Inorder traversal of the constructed AVL tree:
10 20 25 30 40 50 

Explanation:
	•	The AVL Tree remains balanced after each insertion.
	•	For the given keys, the AVL tree will perform necessary rotations to maintain its balance property.

Advantages of AVL Trees:
	1.	Balanced Tree: Guarantees ￼ height, which ensures efficient search, insertion, and deletion.
	2.	Fast Lookups: Because the tree is balanced, AVL trees are efficient for scenarios where frequent lookups are required.

Disadvantages:
	•	Rotations Overhead: Insertion and deletion may require one or more rotations, which adds some overhead compared to a standard BST.
	•	Complex Implementation: Requires more complex code for insertion, deletion, and maintaining balance.

Use Cases:
	•	Databases: AVL Trees are used in database systems where fast retrieval and balanced storage are critical.
	•	Memory Management: They are also used in memory management systems for allocating and deallocating memory blocks efficiently.

AVL Trees are a critical data structure for situations where balanced performance for dynamic insertions, deletions, and searches is necessary. They ensure a balanced height to maintain efficient performance across operations.


---
## **Red-Black Trees**
Certainly! Here is the Red-Black Trees section:

Red-Black Trees (Chapter 13: Balanced Trees)

Red-Black Trees are self-balancing binary search trees that ensure the tree remains balanced during insertions and deletions, maintaining efficient time complexity for search, insert, and delete operations. They are widely used in practice due to their properties that enforce balanced tree height while keeping the operations relatively straightforward.

Key Properties of Red-Black Trees:
	1.	Node Color: Every node is either red or black.
	2.	Root Property: The root is always black.
	3.	Red Node Property: Red nodes cannot have red children (no two consecutive red nodes).
	4.	Black Height Property: Every path from a given node to its descendant null nodes must have the same number of black nodes.
	5.	Leaf Nodes: All leaf nodes (i.e., null nodes) are considered black.

The properties of red-black trees ensure that the height of the tree is logarithmic in terms of the number of nodes, resulting in efficient time complexity for key operations:
	•	Insertion: ￼
	•	Deletion: ￼
	•	Search: ￼

Rotations and Recoloring:
	•	Rotations: Just like AVL trees, red-black trees use left and right rotations to maintain balance. They use rotations to restore balance after inserting or deleting nodes.
	•	Recoloring: When adding or removing nodes, the color of nodes may need to be adjusted to maintain the red-black properties.

Insertion in Red-Black Trees:

The insertion operation in red-black trees is similar to insertion in binary search trees. However, after inserting a new node, the tree must be rebalanced to maintain the red-black properties. This can be achieved by a series of rotations and recoloring.

Python Implementation:

Red-Black Tree Node Class:

class Node:
    def __init__(self, key):
        self.key = key
        self.color = "RED"  # New nodes are always red initially
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    def __init__(self):
        self.NIL_LEAF = Node(key=None)
        self.NIL_LEAF.color = "BLACK"
        self.root = self.NIL_LEAF

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL_LEAF:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.NIL_LEAF:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    def insert(self, key):
        new_node = Node(key)
        new_node.left = self.NIL_LEAF
        new_node.right = self.NIL_LEAF
        new_node.parent = None

        y = None
        x = self.root

        while x != self.NIL_LEAF:
            y = x
            if new_node.key < x.key:
                x = x.left
            else:
                x = x.right

        new_node.parent = y
        if y is None:
            self.root = new_node
        elif new_node.key < y.key:
            y.left = new_node
        else:
            y.right = new_node

        new_node.color = "RED"
        self.insert_fix(new_node)

    def insert_fix(self, k):
        while k != self.root and k.parent.color == "RED":
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right  # uncle node
                if u.color == "RED":
                    # Case 1: Uncle is red
                    k.parent.color = "BLACK"
                    u.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # Case 2: Uncle is black and k is a right child
                        k = k.parent
                        self.left_rotate(k)
                    # Case 3: Uncle is black and k is a left child
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    self.right_rotate(k.parent.parent)
            else:
                u = k.parent.parent.left  # uncle node
                if u.color == "RED":
                    # Case 1: Uncle is red
                    k.parent.color = "BLACK"
                    u.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # Case 2: Uncle is black and k is a left child
                        k = k.parent
                        self.right_rotate(k)
                    # Case 3: Uncle is black and k is a right child
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    self.left_rotate(k.parent.parent)
        self.root.color = "BLACK"

Example Usage:

if __name__ == "__main__":
    rb_tree = RedBlackTree()
    keys = [10, 20, 30, 15, 25, 5]

    for key in keys:
        rb_tree.insert(key)

    print("Red-Black Tree constructed successfully.")

Example Output:

Red-Black Tree constructed successfully.

Properties of Red-Black Trees:

	•	The root is always black.
	•	Red nodes must have black children, which ensures no two consecutive red nodes.
	•	The number of black nodes is the same for every path from the root to the leaf, which ensures the balanced height of the tree.

Advantages of Red-Black Trees:

	•	Balancing Guarantees: By maintaining height balance, red-black trees provide guarantees on the worst-case time complexity for search, insertion, and deletion operations (￼).
	•	Rotations Are Less Frequent: Compared to AVL trees, red-black trees perform fewer rotations on average, making them more efficient for scenarios with frequent insertions and deletions.

Disadvantages:

	•	Slower Lookup Compared to AVL Trees: AVL trees are more strictly balanced, which means that AVL trees may provide faster lookups compared to red-black trees.

Use Cases:

	•	Java TreeMap and TreeSet: Red-Black Trees are used in Java’s TreeMap and TreeSet classes, which provide efficient ordering of data.
	•	Linux Kernel: Red-Black Trees are used in the Linux kernel for managing processes and other system-level data.
	•	Databases: Red-Black Trees are used in some database indexing systems where fast insertions and deletions are needed.

Red-Black Trees are widely used due to their ability to maintain balance with relatively low overhead compared to other balanced tree types, such as AVL Trees. The balance ensures logarithmic height, leading to efficient search, insertion, and deletion operations.


---
## **Flow Algorithms**
---
## **Ford-Fulkerson Algorithm**
Certainly! Here is the Ford-Fulkerson Algorithm section:

Ford-Fulkerson Algorithm (Chapter 26: Flow Networks)

The Ford-Fulkerson Algorithm is used to compute the maximum flow in a flow network. It is based on the idea of finding augmenting paths from the source to the sink and augmenting the flow until no more augmenting paths are available. The algorithm makes use of a residual graph to track possible augmenting paths.

Flow Network:
A flow network is a directed graph where each edge has a capacity and each edge receives a flow that cannot exceed the capacity. There are two nodes of interest:
	1.	Source (s): The starting point for the flow.
	2.	Sink (t): The endpoint for the flow.

Key Terms:
	•	Residual Graph: A graph that shows the remaining capacities for each edge after accounting for the current flow.
	•	Augmenting Path: A path from the source to the sink in the residual graph, where all edges have positive capacity.

The Ford-Fulkerson Algorithm uses a greedy approach to find augmenting paths repeatedly until the maximum possible flow is achieved.

Steps:
	1.	Initialize the flow in all edges to 0.
	2.	While there is an augmenting path from the source to the sink, update the flow along this path.
	3.	Repeat until no more augmenting paths are found.

The time complexity of the algorithm depends on the method used to find augmenting paths. In the worst case, it can be ￼, where ￼ is the number of edges.

Python Implementation:

Here is an implementation of the Ford-Fulkerson Algorithm using Depth First Search (DFS) to find augmenting paths.

Python Code:

from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(lambda: defaultdict(int))

    # Add an edge to the graph
    def add_edge(self, u, v, capacity):
        self.graph[u][v] = capacity

    # A DFS based function to find if there is a path from source 's' to sink 't'
    # in residual graph. Also fills parent[] to store the path
    def dfs(self, s, t, parent):
        visited = [False] * self.V
        stack = [s]

        while stack:
            u = stack.pop()

            for v, capacity in self.graph[u].items():
                if not visited[v] and capacity > 0:  # If not yet visited and capacity is positive
                    stack.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == t:
                        return True
        return False

    # Function to implement the Ford-Fulkerson algorithm
    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.V
        max_flow = 0

        while self.dfs(source, sink, parent):
            # Find the maximum flow through the path found by DFS
            path_flow = float('Inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Update residual capacities of the edges and reverse edges along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

# Example Usage
if __name__ == "__main__":
    g = Graph(6)
    g.add_edge(0, 1, 16)
    g.add_edge(0, 2, 13)
    g.add_edge(1, 2, 10)
    g.add_edge(1, 3, 12)
    g.add_edge(2, 1, 4)
    g.add_edge(2, 4, 14)
    g.add_edge(3, 2, 9)
    g.add_edge(3, 5, 20)
    g.add_edge(4, 3, 7)
    g.add_edge(4, 5, 4)

    source = 0
    sink = 5

    print(f"The maximum possible flow is {g.ford_fulkerson(source, sink)}")

Example Output:

The maximum possible flow is 23

Explanation:
	•	Initialization: The graph is initialized, and the flow is set to 0 for all edges.
	•	DFS to Find Augmenting Path: The algorithm finds an augmenting path from the source (0) to the sink (5). If a path is found, it determines the maximum flow along that path and updates the capacities in the residual graph.
	•	Repeat Until No More Augmenting Paths: The process continues until no more augmenting paths can be found, indicating that the maximum flow has been reached.

Key Concepts:
	1.	Residual Graph: After each iteration, the residual capacities of the edges are updated, and the flow is augmented along the found path.
	2.	Flow Augmentation: Each time an augmenting path is found, the flow is increased by the minimum capacity along that path.
	3.	Termination: The algorithm terminates when no more augmenting paths are available in the residual graph.

Advantages of Ford-Fulkerson:

	•	Simple Implementation: The algorithm is conceptually simple and can be implemented using standard graph traversal techniques such as DFS or BFS.
	•	Versatile: The Ford-Fulkerson method works for any graph as long as there are no negative-weight cycles.

Disadvantages:

	•	Inefficiency with Floating-Point Capacities: The algorithm may enter an infinite loop if the capacities are irrational numbers. This can be resolved by using capacities that are integers.
	•	Slow in Worst Case: The performance depends on the number of augmenting paths needed to reach the maximum flow, and it can be inefficient if the flow increment is very small in each iteration.

Use Cases:

	•	Network Routing: Ford-Fulkerson is used to determine the maximum capacity for data to flow between nodes in a network.
	•	Bipartite Matching: It can be used to find the maximum matching in bipartite graphs, which is useful in job assignment problems.
	•	Circulation Problems: Useful for problems involving the circulation of goods, money, or resources in a network.

The Ford-Fulkerson Algorithm provides a foundation for more advanced maximum flow algorithms such as Edmonds-Karp, which uses BFS for finding augmenting paths to ensure that the solution is found in polynomial time.

Let me know if you need more information or examples regarding the Ford-Fulkerson Algorithm or any other flow algorithms!
---
## **Miscellaneous Algorithms**
---
## **Union-Find**
Certainly! Here is the **Union-Find** section:

---

### **Union-Find (Chapter 21: Data Structures for Disjoint Sets)**
Union-Find, also known as the Disjoint Set Union (DSU), is a data structure that keeps track of a set of elements partitioned into disjoint (non-overlapping) subsets. It provides efficient operations for merging sets (union) and finding the representative of a set (find).

**Steps**:
1. **Find**: Determines which subset a particular element is in. It is also used to check if two elements are in the same subset.
2. **Union**: Merges two subsets into a single subset.

**Path Compression** is used in the **Find** operation to make the tree flat, thereby improving efficiency. **Union by Rank** is used to keep the tree shallow during the **Union** operation.

**Python Implementation**:
```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
```

**Example Usage**:
```python
if __name__ == "__main__":
    uf = UnionFind(5)

    # Perform unions
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)

    # Find representatives
    print(uf.find(0))  # Should print the representative of the set containing 0, 1, 2
    print(uf.find(3))  # Should print the representative of the set containing 3, 4
    print(uf.find(4))  # Should also print the representative of the set containing 3, 4
```

**Example Output**:
```
0
3
3
```

**Explanation**:
- After performing `union(0, 1)` and `union(1, 2)`, elements 0, 1, and 2 are all part of the same set.
- The `union(3, 4)` operation combines elements 3 and 4 into a different set.
- The `find()` operation helps to determine the root representative of each element, ensuring that all connected elements have the same representative.


---
## **How to Use**
Clone this repository and run the individual Python files to see how each algorithm works. You can modify the input arrays to test different scenarios.

```sh
git clone https://github.com/your_username/introduction_to_algorithms.git
cd introduction_to_algorithms
python3 insertion_sort.py
```

---

## **Contributing**
Feel free to open issues or submit pull requests if you have suggestions or improvements. Contributions are always welcome!

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




