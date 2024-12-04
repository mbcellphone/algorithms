#!/usr/bin/env python3

"""
Module Name: greedy_algorithms.py
Description: Examples for Greedy Algorithms (Activity Selection, Huffman Coding)
Author: Marvin Billings
Date: 12/02/2024
"""

import heapq
from collections import Counter, defaultdict

# Activity Selection Problem
def activity_selection(start_times, end_times):
    """
    Solves the activity selection problem to find the maximum number of activities that can be performed.
    :param start_times: List of start times for activities.
    :param end_times: List of end times for activities.
    :return: List of selected activities.
    """
    activities = sorted(list(zip(start_times, end_times)), key=lambda x: x[1])
    selected_activities = []
    last_end_time = 0
    
    for start, end in activities:
        if start >= last_end_time:
            selected_activities.append((start, end))
            last_end_time = end
    
    return selected_activities

# Huffman Coding Implementation
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(data):
    """
    Builds the Huffman Tree and generates the Huffman codes for characters.
    :param data: The input string to encode.
    :return: A dictionary of characters and their corresponding Huffman codes.
    """
    if not data:
        return {}
    
    frequency = Counter(data)
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)
    
    root = priority_queue[0]
    huffman_codes = {}
    _generate_huffman_codes(root, "", huffman_codes)
    return huffman_codes

def _generate_huffman_codes(node, current_code, huffman_codes):
    """
    Helper function to generate Huffman codes by traversing the Huffman Tree.
    :param node: The current node in the Huffman Tree.
    :param current_code: The current Huffman code being generated.
    :param huffman_codes: Dictionary to store the final codes.
    """
    if node is None:
        return
    
    if node.char is not None:
        huffman_codes[node.char] = current_code
    
    _generate_huffman_codes(node.left, current_code + "0", huffman_codes)
    _generate_huffman_codes(node.right, current_code + "1", huffman_codes)

# Main execution
if __name__ == "__main__":
    # Test Activity Selection Problem
    print("Testing Activity Selection Problem")
    start_times = [1, 3, 0, 5, 8, 5]
    end_times = [2, 4, 6, 7, 9, 9]
    selected_activities = activity_selection(start_times, end_times)
    print("Selected activities:", selected_activities)
    print()

    # Test Huffman Coding
    print("Testing Huffman Coding")
    data = "huffman coding is a compression algorithm"
    huffman_codes = huffman_encoding(data)
    print("Huffman Codes:")
    for char, code in huffman_codes.items():
        print(f"'{char}': {code}")
    print()

