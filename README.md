# algorithms
Example: Merge Sort (Chapter 2: Divide-and-Conquer)

Merge Sort is a classic algorithm that divides an array into halves, recursively sorts them, and merges the sorted halves.

Merge Sort Algorithm (from the book)

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


# Example usage
if __name__ == "__main__":
    array = [38, 27, 43, 3, 9, 82, 10]
    print("Original array:", array)
    merge_sort(array)
    print("Sorted array:", array)

Let’s implement Insertion Sort from Chapter 2. This is one of the simplest sorting algorithms. It works by building the sorted portion of the array one element at a time.

Insertion Sort Algorithm (from the book)

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


# Example usage
if __name__ == "__main__":
    array = [12, 11, 13, 5, 6]
    print("Original array:", array)
    insertion_sort(array)
    print("Sorted array:", array)

Explanation

	1.	Outer Loop: Iterates over the unsorted portion of the array starting from the second element.
	2.	Inner Loop: Moves larger elements of the sorted portion one step to the right to make space for the current element (key).
	3.	Insert: Places the key in its correct position.

Example Execution

Input:

array = [12, 11, 13, 5, 6]

Output:

Original array: [12, 11, 13, 5, 6]
Sorted array: [5, 6, 11, 12, 13]

Let me know if you’d like to optimize this further, compare it to other sorting methods, or move on to the next algorithm!
