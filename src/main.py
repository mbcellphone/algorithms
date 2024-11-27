#!/usr/bin/env python3

"""
Module Name: main.py
Description: Examples for Introduction to Algorithms
Author: Marvin Billings
Date: 11/27/2024
"""

# Merge Sort Function
def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    if len(arr) > 1:
        # Find the middle point
        mid = len(arr) // 2
        
        # Divide the array into two halves
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        # Recursively sort both halves
        merge_sort(left_half)
        merge_sort(right_half)
        
        # Merge the sorted halves
        i = j = k = 0
        
        # Copy data to temp arrays left_half and right_half
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        
        # Check for any remaining elements
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


# Insertion Sort Function
def insertion_sort(arr):
    """
    Sorts an array using the Insertion Sort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    # Traverse through elements 1 to n-1
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements that are greater than key to one position ahead
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Place the key in its correct position
        arr[j + 1] = key


# Main execution
if __name__ == "__main__":
    # Test Merge Sort
    print("Testing Merge Sort")
    array = [38, 27, 43, 3, 9, 82, 10]
    print("Original array:", array)
    merge_sort(array)
    print("Sorted array:", array)

    # Test Insertion Sort
    print("\nTesting Insertion Sort")
    array = [12, 11, 13, 5, 6]
    print("Original array:", array)
    insertion_sort(array)
    print("Sorted array:", array)