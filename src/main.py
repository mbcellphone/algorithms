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

def quicksort(arr):
    """
    Sorts an array using the Quicksort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    def partition(low, high):
        # Select the last element as the pivot
        pivot = arr[high]
        i = low - 1  # Pointer for the smaller element
        
        for j in range(low, high):
            # If the current element is smaller than or equal to the pivot
            if arr[j] <= pivot:
                i += 1  # Increment the index of the smaller element
                arr[i], arr[j] = arr[j], arr[i]  # Swap
        
        # Place the pivot element at the correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_helper(low, high):
        if low < high:
            # Partition the array and get the pivot index
            pivot_index = partition(low, high)
            
            # Recursively apply to the left and right subarrays
            quicksort_helper(low, pivot_index - 1)
            quicksort_helper(pivot_index + 1, high)
    
    quicksort_helper(0, len(arr) - 1)




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
    

    # Test Quicksort
    print("Testing Quicksort")
    array = [10, 7, 8, 9, 1, 5]
    print("Original array:", array)
    quicksort(array)
    print("Sorted array:", array)