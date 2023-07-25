import random
import time
def generate_prime_number(bit_length):
    while True:
    prime_candidate = random.getrandbits(bit_length)
    if is_prime(prime_candidate):
    return prime_candidate
def is_prime(n, k=5):
    if n <= 3:
    return n == 2 or n == 3
    if n % 2 == 0:
    return False
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1 
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
    for _ in range(r - 1):
    x = pow(x, 2, n)
 if x == n - 1:
 break
 else:
 return False
 return True
def generate_dh_key_pair(prime, generator):
 private_key = random.randint(2, prime - 2)
 public_key = pow(generator, private_key, prime)
 return private_key, public_key
2
def generate_shared_secret(private_key, other_public_key, prime):
 shared_secret = pow(other_public_key, private_key, prime)
 return shared_secret
# Example usage
bit_length = 1000 # Specify the desired bit length here
start_time=time.time()
# Generate prime number
prime = generate_prime_number(bit_length)
# Choose a generator (commonly 2 or 5)
generator = random.randint(2, prime - 1)
# Generate Diffie-Hellman key pair
private_key, public_key = generate_dh_key_pair(prime, generator)
# Example of other party's public key (pretending it was received)
other_public_key = 123456789 # Replace with the actual other party's public key
# Generate shared secret
shared_secret = generate_shared_secret(private_key, other_public_key, prime)
end_time=time.time()
print("Prime:", prime)
print("Generator:", generator)
print("Private Key:", private_key)
print("Public Key:", public_key)
print("Shared Secret:", shared_secret)
print("Time taken",end_time-start_time) 