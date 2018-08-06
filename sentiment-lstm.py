# emotion analysys by RNN 
import numpy as np
import tensorflow as tf
with open('./reviews.txt','r') as f: # open this directory file as readonly and make referable as f
    reviews = f.read() # f.read() substitute contents of f for reviews

# print(reviews[:200]) # slice : pick up 0~199 data
with open('./labels.txt','r') as f:
    labels = f.read()

# print(labels) 

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text=' '.join(reviews) # Afrte connect all text, separate words by space.
words = all_text.split()

# print(all_text) 

print(reviews)


# print(words)