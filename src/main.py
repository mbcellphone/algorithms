#!/usr/bin/env python3

"""
Module Name: main.py
Description: Examples for Introductions to Algorithms 
Author: Marvin Billings
Date: 11/27/2024
"""


#Merge Sort 
#Chapter2: Divide-and-Conquer
# Divide the array elements into two halves
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
