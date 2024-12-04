#!/usr/bin/env python3

"""
Module Name: mathematical_algorithms.py
Description: Examples for Mathematical Algorithms (Greatest Common Divisor, Sieve of Eratosthenes, Modular Exponentiation)
Author: Marvin Billings
Date: 12/02/2024
"""

# Greatest Common Divisor (GCD) using Euclidean Algorithm
def gcd(a, b):
    """
    Finds the Greatest Common Divisor of two numbers using the Euclidean algorithm.
    :param a: First number.
    :param b: Second number.
    :return: GCD of a and b.
    """
    while b != 0:
        a, b = b, a % b
    return a

# Sieve of Eratosthenes to find all primes up to a given limit
def sieve_of_eratosthenes(limit):
    """
    Finds all prime numbers up to a given limit using the Sieve of Eratosthenes.
    :param limit: The upper limit to find primes.
    :return: List of prime numbers up to the limit.
    """
    primes = [True] * (limit + 1)
    p = 2
    while (p * p <= limit):
        if primes[p]:
            for i in range(p * p, limit + 1, p):
                primes[i] = False
        p += 1
    
    return [p for p in range(2, limit + 1) if primes[p]]

# Modular Exponentiation
# This function is used to perform (base^exp) % mod efficiently
def modular_exponentiation(base, exp, mod):
    """
    Performs modular exponentiation.
    :param base: Base number.
    :param exp: Exponent.
    :param mod: Modulus.
    :return: Result of (base^exp) % mod.
    """
    result = 1
    base = base % mod
    while exp > 0:
        if (exp % 2) == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

# Main execution
if __name__ == "__main__":
    # Test GCD using Euclidean Algorithm
    print("Testing GCD (Greatest Common Divisor)")
    a, b = 56, 98
    print(f"GCD of {a} and {b}: {gcd(a, b)}")
    print()

    # Test Sieve of Eratosthenes
    print("Testing Sieve of Eratosthenes")
    limit = 50
    print(f"Prime numbers up to {limit}: {sieve_of_eratosthenes(limit)}")
    print()

    # Test Modular Exponentiation
    print("Testing Modular Exponentiation")
    base = 3
    exp = 13
    mod = 17
    print(f"({base}^{exp}) % {mod} = {modular_exponentiation(base, exp, mod)}")
    print()

