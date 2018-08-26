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

# print("Train set: \t\t{}".format(train_x.shape))

# print("Validation set: \t{}".format(val_x.shape))

# print("Test set: \t{}".format(test_x.shape))


# graph definition

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001

n_words = len(vocab_to_int) + 1

graph = tf.Graph()
with graph.as_default():
    inputs_=tf.placeholder(tf.int32,[None,None],name='inputs')
    labels_=tf.placeholder(tf.int32,[None,None],name='labels')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

embed_size=300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-1,1))
    embed = tf.nn.embedding_lookup(embedding, inputs_) # if you input embedding input-value, return word vector


# define LSTM Cell and Layer

with graph.as_default():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers) # results of drop*lstm_layer 
    initial_state = cell.zero_state(batch_size,tf.float32) # initalize cell

    # def of output
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell,embed,initial_state=initial_state)

# prediction
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:,-1],1,activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_,predictions)  
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # calc of learning accuracy

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions),tf.int32),labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# make batch module 
def get_batches(x,y,batch_size=100):
    n_batches = len(x)//batch_size
    x,y = x[:n_batches*batch_size],y[:n_batches*batch_size]
    for ii in range(0,len(x), batch_size):
        yield x[ii:ii+batch_size],y[ii:ii*batch_size]

# training 
epochs = 10
with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        for ii, (x,y) in enumerate(get_batches(train_x,train_y,batch_size),1):
            feed ={inputs_: x,
                    labels_:y[:,None],
                    keep_prob:0.5,
                    initial_state:state}
            loss,state,_ = sess.run([cost,final_state,optimizer],feed_dict=feed)

            if iteration%5==0:
                print("Epoch:{}/{}".format(e, epochs),
                     "Iteration:{}".format(iteration),
                     "Training Loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size,tf.float32))
                for x,y in get_batches(val_x,val_y,batch_size):
                    feed = {inputs_ :x,
                            labels_:y[:,None],
                            keep_prob:1, # No Dropout
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state],feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Value Acc: {:.3f}".format(np.mean(val_acc)))
            iteration+=1
    saver.save(sess,"checkpoint/sentiment.ckpt")
