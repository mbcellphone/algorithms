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
---
### **Kruskal's Algorithm**
---
### **Prim's Algorithm**
---
## **Dynamic Programming Algorithms**
---
### **Fibonacci Sequence**
---
## **Longest Common Subsequence (LCS)**
---
## **Knapsack Problem**
---
## **Greedy Algorithms**
---
## **Activity Selection Problem**
---
## **Huffman Coding**
---
## **Divide-and-Conquer Algorithms**
---
## **Strassen's Algorithm**
---
## **Backtracking Algorithms**
---
## **N-Queens Problem**
---
## **Sudoku Solver**
---
## **Mathematical Algorithms**
---
## **GCD (Euclidean Algorithm)**
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
## **
AVL Trees**
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




