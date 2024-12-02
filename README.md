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
---
## **String Algorithms**
---
## **Rabin-Karp Algorithm**
---
## **Knuth-Morris-Pratt (KMP) Algorithm**
---
## **Tree Algorithms**
---
## **Binary Search Tree (BST) Operations**
---
## **AVL Trees**
---
## **Red-Black Trees**
---
## **Flow Algorithms**
---
## **Ford-Fulkerson Algorithm**
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




