#!/usr/bin/env python3

"""
Module Name: probabilistic_randomized_algorithms.py
Description: Examples for Probabilistic and Randomized Algorithms (Monte Carlo Method, Reservoir Sampling, Randomized QuickSort)
Author: Marvin Billings
Date: 12/02/2024
"""

import random

# Monte Carlo Method to Estimate Pi
def monte_carlo_pi(num_samples):
    """
    Estimates the value of Pi using the Monte Carlo method.
    :param num_samples: Number of random points to generate.
    :return: Estimated value of Pi.
    """
    inside_circle = 0
    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x ** 2 + y ** 2 <= 1:
            inside_circle += 1
    
    return (inside_circle / num_samples) * 4

# Reservoir Sampling Algorithm
# This algorithm is used to select k items from a stream of n items, where n is large or unknown
# This implementation selects 1 item from a stream (k = 1)
def reservoir_sampling(stream):
    """
    Performs reservoir sampling to randomly select one item from a stream.
    :param stream: Iterable representing the stream of data.
    :return: A randomly selected item from the stream.
    """
    sample = None
    for i, item in enumerate(stream, start=1):
        if random.randint(1, i) == 1:
            sample = item
    return sample

# Randomized QuickSort Algorithm
def randomized_partition(arr, low, high):
    """
    Partitions the array for QuickSort by selecting a random pivot.
    :param arr: List of elements to sort.
    :param low: Starting index.
    :param high: Ending index.
    :return: Index of the pivot element.
    """
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def randomized_quicksort(arr, low, high):
    """
    Sorts an array using the Randomized QuickSort algorithm.
    :param arr: List of elements to sort.
    :param low: Starting index.
    :param high: Ending index.
    :return: None (sorts the array in place).
    """
    if low < high:
        pivot_index = randomized_partition(arr, low, high)
        randomized_quicksort(arr, low, pivot_index - 1)
        randomized_quicksort(arr, pivot_index + 1, high)

# Main execution
if __name__ == "__main__":
    # Test Monte Carlo Method to Estimate Pi
    print("Testing Monte Carlo Method to Estimate Pi")
    num_samples = 10000
    estimated_pi = monte_carlo_pi(num_samples)
    print(f"Estimated value of Pi using {num_samples} samples: {estimated_pi}")
    print()

    # Test Reservoir Sampling
    print("Testing Reservoir Sampling")
    stream = range(1, 101)  # Stream of numbers from 1 to 100
    sampled_item = reservoir_sampling(stream)
    print(f"Randomly selected item from the stream: {sampled_item}")
    print()

    # Test Randomized QuickSort
    print("Testing Randomized QuickSort")
    array = [10, 7, 8, 9, 1, 5]
    print("Original array:", array)
    randomized_quicksort(array, 0, len(array) - 1)
    print("Sorted array:", array)
    print()

