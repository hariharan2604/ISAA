import random
import string

def random_string(length):
    chars=string.ascii_letters+string.digits+string.punctuation
    result=''.join(random.choice(chars) for _ in range(length))
    return result

my_string=input("Enter the string :")
print("Input String is ",my_string)
print("The string split:",end="")
length=random.randint(0,len(my_string)-1)
split1=my_string[0:length]
print("A:",split1)
split1=split1.encode("ascii")
split2=my_string[length:]
print("B:",split2)
split2=split2.encode("ascii")
decoded_string=""
key=random_string(2000)
key=key.encode("ascii")
print("The key generated :",key)
print("The encrypted data :",end="")

encoded_word1=bytes([a^b for a,b in zip(split1,key)])
print("Cipher1 :",encoded_word1.decode("ascii"))

encoded_word2=bytes([a^b for a,b in zip(split2,key)])
print("Cipher2 :",encoded_word2.decode("ascii"))


decoded_word1=bytes([a^b for a,b in zip(encoded_word1,key)])
decoded_string+=decoded_word1.decode("ascii")

decoded_word2=bytes([a^b for a,b in zip(encoded_word2,key)])
decoded_string+=decoded_word2.decode("ascii")

print("The decoded string is ",decoded_string)