# Introduction to Algorithms
Example: Merge Sort (Chapter 2: Divide-and-Conquer)

Merge Sort is a classic algorithm that divides an array into halves, recursively sorts them, and merges the sorted halves.

## Merge Sort Algorithm (from the book)

	•	Input: Array  A[p..r] 
	•	Output: Sorted array  A[p..r] 
	•	Divide the array into two halves.
	•	Recursively sort the subarrays.
	•	Merge the sorted subarrays.

Here’s the Python implementation:

def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    if len(arr) > 1:
        # Find the middle point
        mid = len(arr) // 2
        
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


### Example usage
if __name__ == "__main__":
    array = [38, 27, 43, 3, 9, 82, 10]
    print("Original array:", array)
    merge_sort(array)
    print("Sorted array:", array)


## Insertion Sort Algorithm from Chapter 2

	•	Iterate through each element of the array.
	•	Compare the current element with the elements in the sorted portion (to the left).
	•	Shift elements of the sorted portion to the right until the correct position for the current element is found.
	•	Insert the current element into its correct position.

Python Implementation

def insertion_sort(arr):
    """
    Sorts an array using the Insertion Sort algorithm.
    :param arr: List of numbers to sort.
    :return: None (in-place sorting).
    """
    # Traverse from the second element to the end
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements of arr[0..i-1] that are greater than key
        # to one position ahead of their current position
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Place the key in its correct position
        arr[j + 1] = key


### Example usage
if __name__ == "__main__":
    array = [12, 11, 13, 5, 6]
    print("Original array:", array)
    insertion_sort(array)
    print("Sorted array:", array)

Explanation

	1.	Outer Loop: Iterates over the unsorted portion of the array starting from the second element.
	2.	Inner Loop: Moves larger elements of the sorted portion one step to the right to make space for the current element (key).
	3.	Insert: Places the key in its correct position.

### Example Execution

Input:

array = [12, 11, 13, 5, 6]

Output:

Original array: [12, 11, 13, 5, 6]
Sorted array: [5, 6, 11, 12, 13]



Let’s move on to the Quicksort algorithm, another popular and efficient sorting method, also from Chapter 7 of the book.

Quicksort Algorithm (from the book)

Quicksort is a divide-and-conquer algorithm that:
	1.	Selects a pivot element.
	2.	Partitions the array such that elements smaller than the pivot are moved to its left, and elements larger are moved to its right.
	3.	Recursively applies the same process to the left and right subarrays.

Python Implementation

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


# Example usage
if __name__ == "__main__":
    # Test Quicksort
    print("Testing Quicksort")
    array = [10, 7, 8, 9, 1, 5]
    print("Original array:", array)
    quicksort(array)
    print("Sorted array:", array)

Explanation

	1.	Partition Function:
	•	Chooses the last element as the pivot.
	•	Rearranges the array so that elements smaller than the pivot come before it, and larger elements go after it.
	•	Returns the index of the pivot after partitioning.
	2.	Recursive Helper Function:
	•	Recursively sorts the left and right subarrays until the entire array is sorted.
	3.	Base Case:
	•	The recursion stops when the low index is no longer less than the high index.

Example Output

Input:

array = [10, 7, 8, 9, 1, 5]

Output:

Testing Quicksort
Original array: [10, 7, 8, 9, 1, 5]
Sorted array: [1, 5, 7, 8, 9, 10]

Let me know if you want to dive into:
	•	Analyzing the time complexity of Quicksort.
	•	Optimizing the pivot selection (e.g., using the median-of-three method).
	•	Moving to the next algorithm, such as Heapsort!

Let me know if you’d like to optimize this further, compare it to other sorting methods, or move on to the next algorithm!
