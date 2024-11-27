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

