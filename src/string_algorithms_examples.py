#!/usr/bin/env python3

"""
Module Name: string_algorithms.py
Description: Examples for String Algorithms (Rabin-Karp Algorithm, Knuth-Morris-Pratt Algorithm)
Author: Marvin Billings
Date: 12/02/2024
"""

# Rabin-Karp Algorithm for Pattern Matching
def rabin_karp(text, pattern, prime=101):
    """
    Implements Rabin-Karp algorithm for string matching.
    :param text: The text in which to search for the pattern.
    :param pattern: The pattern to search for.
    :param prime: A prime number used for hashing.
    :return: Starting index of matches in the text.
    """
    m = len(pattern)
    n = len(text)
    d = 256  # Number of characters in the input alphabet
    p = 0  # Hash value for pattern
    t = 0  # Hash value for text
    h = 1
    
    # The value of h would be "pow(d, m-1) % prime"
    for i in range(m - 1):
        h = (h * d) % prime
    
    # Calculate the hash value of the pattern and first window of text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime
    
    # Slide the pattern over the text one by one
    for i in range(n - m + 1):
        # Check the hash values of current window of text and pattern
        if p == t:
            # If the hash values match, check the characters one by one
            if text[i:i + m] == pattern:
                print(f"Pattern found at index {i}")
        
        # Calculate hash value for next window of text: Remove leading digit, add trailing digit
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % prime
            # We might get negative values of t, converting it to positive
            if t < 0:
                t = t + prime

# Knuth-Morris-Pratt (KMP) Algorithm for Pattern Matching
def kmp_search(text, pattern):
    """
    Implements Knuth-Morris-Pratt (KMP) algorithm for string matching.
    :param text: The text in which to search for the pattern.
    :param pattern: The pattern to search for.
    :return: Starting index of matches in the text.
    """
    m = len(pattern)
    n = len(text)
    lps = [0] * m  # Longest Prefix Suffix array
    j = 0  # Index for pattern[]
    
    # Preprocess the pattern to calculate lps array
    compute_lps_array(pattern, m, lps)
    
    i = 0  # Index for text[]
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            print(f"Pattern found at index {i - j}")
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

def compute_lps_array(pattern, m, lps):
    """
    Helper function to compute the Longest Prefix Suffix (lps) array for KMP algorithm.
    :param pattern: The pattern for which to compute the lps array.
    :param m: Length of the pattern.
    :param lps: Array to store the longest prefix suffix values.
    :return: None (modifies lps array in place).
    """
    length = 0  # Length of the previous longest prefix suffix
    i = 1
    lps[0] = 0  # lps[0] is always 0
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

# Main execution
if __name__ == "__main__":
    # Test Rabin-Karp Algorithm
    print("Testing Rabin-Karp Algorithm")
    text = "ABCCDDAEFG"
    pattern = "CDD"
    rabin_karp(text, pattern)
    print()

    # Test Knuth-Morris-Pratt (KMP) Algorithm
    print("Testing Knuth-Morris-Pratt (KMP) Algorithm")
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    kmp_search(text, pattern)
    print()

