import math

def is_square(n):
    # Helper function to check if a number is a perfect square
    root = int(math.sqrt(n))
    return root * root == n

def fermat_factorization(n):
    # Check if the number is even
    if n % 2 == 0:
        return [2, n // 2]

    # Try to find a square root of n (rounded up to the nearest integer)
    a = math.isqrt(n) + 1
    b2 = a * a - n

    # Continue until b2 is a perfect square
    while not is_square(b2):
        a += 1
        b2 = a * a - n

    # Compute the factors
    b = math.isqrt(b2)
    factor1 = a - b
    factor2 = a + b

    return [factor1, factor2]

# Test the function
if __name__ == "__main__":
    num_to_factorize = 8051
    factors = fermat_factorization(num_to_factorize)
    print(f"The factors of {num_to_factorize} are: {factors}")
