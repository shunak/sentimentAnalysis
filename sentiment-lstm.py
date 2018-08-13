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

# print(reviews)

# print(words)
from collections import Counter
counts = Counter(words)
vocab=sorted(counts, key=counts.get, reverse=True) #get words by desc

# print(vocab)
vocab_to_int = {word:ii for ii,word in enumerate(vocab,1)} # get key and value by enumerate
# print(vocab_to_int) 

reviews_int=[] # convert reviews literal to int
for each in reviews:
    reviews_int.append([vocab_to_int[word] for word in each.split()])

# print(reviews_int)
# convert label to vector 
labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])

# print(labels[:50])
review_lens =Counter([len(x) for x in reviews_int])
# print(review_lens[0]) 
# print(max(review_lens))
non_zero_idx = [ii for ii, review in enumerate(reviews_int) if len(review)!=0]

reviews_int = [reviews_int[ii] for ii in non_zero_idx] # get non zero value
labels = np.array([labels[ii] for ii in non_zero_idx])
# print(len(non_zero_idx))
# print(reviews_int[-1])
seq_len = 200
features = np.zeros((len(reviews_int), seq_len),dtype=int)

for i, row in enumerate(reviews_int):
    features[i,-len(row):]=np.array(row)[:seq_len]


# print(features[:10,:100])
# split data for training and validation and test 

split_frac = 0.8
split_idx = int(len(features)*0.8) 


train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y =labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)

val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:] 

print("Train set: \t\t{}".format(train_x.shape))

print("Validation set: \t{}".format(val_x.shape))

print("Test set: \t{}".format(test_x.shape))


# graph definition

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001

n_words = len(vocab_to_int) + 1

graph = tf.Graph()
with graph.as_default():
    inputs_=tf.placeholder(tf.int32,[None,None],name='inputs')
    labels_=tf.placeholder(tf,int32,[None,None],name='labels')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

embed_size=300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-1,1))
    embed = tf.nn.embedding_lookup(embedding, inputs_) # if you input embedding input-value, return word vector












