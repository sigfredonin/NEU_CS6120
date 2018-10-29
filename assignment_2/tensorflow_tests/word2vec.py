"""
word2vec
Author: Nishith Khandwala (nishith@stanford.edu)
Adapted from https://www.tensorflow.org/tutorials/word2vec

Taken from You-Tube video ...
Stanford CS224n Natural Language Processing with Deep Learning
Lecture 7: Introduction to TensorFlow
https://youtu.be/PicxU81owCs
See also committed code on github --
    https://github.com/nishithbsk/tensorflow_tutorials/blob/master/cs224n
    * word2vec_complete.py
    * utils.py
    * get_data.py

Changes:
Omitted:    imports from __future__ not needed for Python 3
Fixed:  inputs=batch_inputs -> inputs=batch_embeddings
        in tf.nn.nce_loss() call
Fixed:  keep_dims=True -> keepdims=True
        in tf.reduce_sum() (and reduce_mean())
        keep_dims is deprecated, replaced by keep_dims
Fixed:  tf.reduce_mean() -> tf.reduce_sum() in norm calculation,
        to match committed code on github (after the fact)
        It works with tf.reduce_mean(), maybe not as well.
Adapted: Use 'step + 1' to decide when to print progress,
        to account for counting from zero.
Added:  tracing to monitor execution progress

Sig Nin
October 24, 2018
"""
# .. for Python 2.7 ...
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

from utils import *

'''
Consider the following sentence:
"the first cs224n homework was a lot of fun"

With a window of size 1, we have the dataset:
([the, cs224n], first), ([lot, fun], of), ...

Remember that Skipgram tries to predict each context word from
its target word, and so the task becomes to predict 'the' and
'cs224n' from 'first', and 'lot and 'fun' from 'of', and so on

Our dataset now becomes:
(first, the), (first, cs224n), (of, lot), (of, fun), ...
'''

# Let's defind some constants first
batch_size = 128
vocabulary_size = 50000
embedding_size = 128 # Dimensions of the embedding vector.
num_sampled = 64 # Number of negative examples to sample

'''
load_data() loads the already preprocessed training and val data.

train_data is a list of (batch_input, batch_labels) pairs.
val_data is a list of all validation inputs.
reverse_dictionary is a python dict from word index to word.
'''
print("Load data ...")
train_data, val_data, reverse_dictionary = load_data()
print("Number of training examples:", len(train_data)*batch_size)
print("Number of validation examples:", len(val_data))
print("... data loaded.")

def skipgram():
    batch_inputs = tf.placeholder(tf.int32, shape=[batch_size,])
    batch_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    val_dataset = tf.constant(val_data, dtype=tf.int32)

    with tf.variable_scope('word2vec') as scope:
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                                                    embedding_size],
                                                    -1.0, 1.0))
        batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)

        weights = tf.Variable(tf.truncated_normal([vocabulary_size,
                                                   embedding_size],
                                                   stddev = 1.0/math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
                                             biases=biases,
                                             labels=batch_labels,
                                             inputs=batch_embeddings,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings/norm
        print("embeddings: %s" % embeddings[0])
        print("norm: %s" % norm)
        print("normalized_embeddings: %s" % normalized_embeddings[0])

        val_embeddings = tf.nn.embedding_lookup(normalized_embeddings, val_dataset)
        print("val_embeddings: %s" % val_embeddings[0])
        similarity = tf.matmul(val_embeddings, normalized_embeddings, transpose_b=True)
        return batch_inputs, batch_labels, normalized_embeddings, loss, similarity

def run():
    print("Begin run ----------")
    batch_inputs, batch_labels, normalized_embeddings, loss, similarity = skipgram()
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("-- variables initialized")

        average_loss = 0.0
        for step, batch_data in enumerate(train_data):
            inputs, labels = batch_data
            feed_dict = {batch_inputs: inputs, batch_labels: labels}

            _, loss_val = sess.run([optimizer, loss], feed_dict)
            average_loss += loss_val

            if (step + 1) % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                print('loss at iter', step + 1, ":", average_loss)
                average_loss = 0.0

            if (step + 1) % 5000 == 0:
                sim = similarity.eval()
                for i in range(len(val_data)):
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    print_closest_words(val_data[i], nearest, reverse_dictionary)

            final_embeddings = normalized_embeddings.eval()
    print("Run ended ----------")

# Let's start training
final_embeddings = run()

# Visualize the embeddings.
visualize_embeddings(final_embeddings, reverse_dictionary)