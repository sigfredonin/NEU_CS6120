"""
utils.py
Utilities to support tensorflow examples

load_data() -
  load preprocessed training and validation data,
  and a reverse dictionary containing the mapping
  of index -> word in the vocabulary
print_closest_words(word, nearest, reverse_dictionary) -
    Print the words closest to the given word.
visualize_embeddings(final_embeddings, reverse_dictionary)
    Plot a projection of the words in relation to one
    another as determined by the calculated embeddings.
    (Placeholder include - not yet implemented.)

Adapted from word2vec_basic in TensorFlow Tutorials.

Sig Nin
October 25, 2018
"""

import tensorflow as tf
import numpy as np
import random
from nltk import FreqDist
from collections import defaultdict
from collections import deque

DEBUG = False

#----------------------------------------------------------------------
# Load words from a clean running text file, like text8 from
# http://mattmahoney.net/dc/text8.zip
#----------------------------------------------------------------------

def get_text_from_file(filePath):
    """
    Read all of the text from a file.
    This is expected to be a file containing the words
    from some text in their original order, with a blank
    between each pair of words, and no punctuation.
    """
    with open(filePath) as f:
        text = f.read()
    return text

def build_dataset(text, vocabulary_size):
    """
    Create the inputs to a TensorFlow NN ---
        data - contains the text encoded as word indices
        vocabulary - the vocabulary words, in index order
        dictionary - a forward mapping, word -> index
        reverse_dictionary - a reverse mapping, index -> word
    """
    # Collect the vocabulary:
    #   'UNK plus the 'vocabulary_size - 1' most frequent
    words = text.split()
    fd = FreqDist(words)
    vocabulary = [('UNK', 0)] + fd.most_common(vocabulary_size - 1)
    # Create a word -> index mapping
    dictionary = {}
    for index, word_count in enumerate(vocabulary):
        word, count = word_count
        dictionary[word] = index
    # Create the data, a list of word indices
    # and count out-of-vocabulary words, to code as 'UNK'
    data = []
    count_UNK = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            count_UNK += 1
        data.append(index)
    vocabulary[0] = ('UNK', count_UNK)
    # Create a reverse mapping, index -> word
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # Show sample of results ...
    print("Text: %s ... %s" % (text[:5], text[-5:]))
    print("Words: %s .. %s" % (words[:5], words[-5:]))
    print("Vocabulary: %s  %s" % (vocabulary[:3], vocabulary[-3:]))
    print("Dictionary: %s ... %s" % \
        (list(dictionary.items())[:3], \
         list(dictionary.items())[-3:]))
    print("Reverse Dictionary: %s ... %s" % \
        (list(reverse_dictionary.items())[:3], \
         list(reverse_dictionary.items())[-3:]))
    print("data: %s ..." % data[:10])

    return data, vocabulary, dictionary, reverse_dictionary

def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    """
    Create an input batch from the data ...
        data - the index-coded word sequence
        data_index - position at which to begin the batch
        batch_size - number of word : label pairs to generate
        num_skips - number of words from context to use per target word
        skip window - number of words before and after target word
    Example:
        ... to see any sense in   ...
            7  68  105 274   5
            context    context
                  target
        skip_window = 2
        num_skips in { 1, 2, 3, 4 }
        batch_size % num_skips = 0      batch_size a multiple of num_skips
        num_skips <= 2 * skip_window    num_skips <= number of context words
                                        (num_skips == 0 does not make sense)
    Output:
        batch:  vector of target words, length batch_size
        labels: vector of context words, length batch_size
        data_index: updated position at which to begin the next batch
    """
    assert data_index < len(data)
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # Define batch, labels, and a buffer to hold the current window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # length of context window
    context_word_indices = [iW for iW in range(span) if iW != skip_window]
    buffer = deque(maxlen=span) # buffer of context window length
                                # when items are added to it, the deque
                                # drops entries from the other end
                                # to stay within maxlen
    # If window extends past the end of the data,
    # set it back to the beginning
    if data_index + span > len(data):
        data_index = 0
    # Collect the first window for this batch
    buffer.extend(data[data_index : data_index + span])
    if DEBUG: print(buffer)
    data_index += span
    # Build the batch data and label data, in blocks of num_skips entries
    batch_data = []
    label_data = []
    for iSB in range(batch_size // num_skips):
        # Choose num_skips context words to use, at random
        words_to_use = random.sample(context_word_indices, num_skips)
        if DEBUG: print("use:", words_to_use, "from", context_word_indices)
        # Collect the chosen context words and add them to the batch
        for iCW in words_to_use:
            batch_data.append(buffer[skip_window]) # target word
            label_data.append(buffer[iCW])         # context word
        # slide the window to its next position (deque keeps its size)
        if data_index < len(data):
            buffer.append(data[data_index])     # move to next target word
            data_index += 1
        else:
            buffer.extend(data[0:span])         # wrap back to beginning
            data_index = span
        if DEBUG: print(buffer)
    assert len(batch_data) == batch_size
    assert len(label_data) == batch_size
    batch[0:] = batch_data
    labels[0:,0] = label_data
    # Backtrack a little bit to avoid skipping words at the end of a batch
    # Goes back one window, but never past the beginning of the data
    data_index = (len(data) + data_index - span) % len(data)
    return batch, labels, data_index

def build_training_batches(data, number_of_batches, \
        batch_size, num_skips, skip_window):
    training_batches = []
    data_index = 0
    for iBatch in range(number_of_batches):
        batch, labels, data_index = \
            generate_batch(data, data_index, batch_size, num_skips, skip_window)
        training_batches.append((batch, labels))
    print("Training batch:", training_batches[0])
    return training_batches

def load_data_from_file(filePath):
    """
    Load the data needed to run the word2vec example.
    Input: file path
    Returns:
        train_data: [ (batch_data, batch_labels), ... ] 30,000 batches
        val_data: [ word_index_0, word_index_1, ... ]   16 random from top 100
        reverse_dictionary: { index : word, ... }
    """
    vocabulary_size = 50000
    number_of_batches = 30000
    batch_size = 128
    num_skips = 4
    skip_window = 2
    val_data_window = 100
    val_data_size = 16

    text = get_text_from_file(filePath)
    data, vocabulary, dictionary, reverse_dictionary = \
        build_dataset(text, vocabulary_size)
    train_data = build_training_batches(data, number_of_batches, \
        batch_size, num_skips, skip_window)
    val_data = np.random.choice(val_data_window, val_data_size, replace=False)
    print("Validation data:", val_data)
    return train_data, val_data, reverse_dictionary

def load_data():
    """
    Load the data needed to run the word2vec example.
        Input file: data/text8
    Returns:
        train_data: [ (batch_data, batch_labels), ... ] 30,000 batches
        val_data: [ word_index_0, word_index_1, ... ]   16 random from top 100
        reverse_dictionary: { index : word, ... }
    """
    filePath = "data/text8"
    return load_data_from_file(filePath)

def print_closest_words(word, nearest, reverse_dictionary):
    """
    Print the words closest to the given word.
    Use the reverse dictionary to decode the word and its nearest neighbors.
    """
    word_str = reverse_dictionary[word]
    nearest_str_list = [ reverse_dictionary[near_word] for near_word in nearest ]
    print("%s: %s" % (word_str, nearest_str_list))

def visualize_embeddings(final_embeddings, reverse_dictionary):
    print("Visualize embeddings?  Maybe later, dude.")
