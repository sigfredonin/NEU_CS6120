import re
import nltk
import numpy as np
from gensim.models import KeyedVectors

DEBUG = False

TRAIN_PATH = "../resources/train_data.txt"
TEST_PATH = "../resources/test_data.txt"
WORD_VECTORS_FILE = "../resources/GoogleNews-vectors-negative300.bin"
THRESHOLD = 1

# tagset from nltl.help.upenn_tagset() + '#' and 'UNK'
TAGSET = ['UNK', '$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', \
          'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
          'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', \
          'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', \
          'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WPS', 'WRB', '``', '#']
TAGS_DICT = { TAGSET[i] : i for i in range(len(TAGSET)) }

# read a file into a string
def file_to_str(filepath):
    with open(filepath) as f:
        file_str = f.read()
    return file_str

# get test tokens (which don't have ratings) and sentences
def test_file_to_tokens(filepath):
    file_str = file_to_str(filepath)
    reviews = re.findall(r'^(.+)$', file_str.lower(), re.MULTILINE)
    tokenized_reviews = [ review.split() for review in reviews ]
    return tokenized_reviews, reviews

# get review tokens, ratings, and overall tokens from a file string
def str_to_tokens_and_ratings(file_str):
    reviews = re.findall(r'^(.+)\|(.+)$', file_str.lower(), re.MULTILINE)
    tokenized_reviews = [ review.split() for review, rating in reviews ]
    review_ratings = [ int(rating) for review, rating in reviews ]
    overall_tokens = [ token for tokenized_review in tokenized_reviews \
                             for token in tokenized_review ]
    return overall_tokens, tokenized_reviews, review_ratings

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ One-Hot Vectors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# find all words with frequency less than threshold
def find_freq(tokens, THRESHOLD):
    fdist = nltk.FreqDist(tokens)
    freq = [word for word in fdist if fdist[word] > THRESHOLD]
    print("frequent token count:", len(freq))
    return freq

# create a dictionary of all the words in the vocabulary to an index
def create_vocab_dict(freq_tokens):
    tokens = ['UNK'] + freq_tokens
    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i
    return vocab_dict

# creates a list of one hot vectors from reviews
def one_hot_vector_reviews(tokenized_reviews, vocab_dict):
    one_hot_vectors = []
    for review in tokenized_reviews:
        one_hot = [0] * len(vocab_dict)
        for token in review:
            one_hot[vocab_dict.get(token, 0)] = 1
        one_hot_vectors.append(one_hot)
    return one_hot_vectors

# loads reviews from a file and tokenizes them
def load_tokens_and_ratings(filepath):
    file_str = file_to_str(filepath)
    overall_tokens, tokenized_reviews, review_ratings = \
        str_to_tokens_and_ratings(file_str)
    return overall_tokens, tokenized_reviews, review_ratings

# converts tokens into a list of one hot vectors
def train_tokens_to_one_hots(overall_tokens, tokenized_reviews):
    freq_tokens = find_freq(overall_tokens, THRESHOLD)
    vocab_dict = create_vocab_dict(freq_tokens)
    one_hot_vectors = one_hot_vector_reviews(tokenized_reviews, vocab_dict)
    return one_hot_vectors, vocab_dict

# converts file into a list of one hot vectors using an existing dictionary
def test_file_to_one_hots(filepath, vocab_dict):
    tokenized_reviews, reviews = test_file_to_tokens(filepath)
    one_hot_vectors = one_hot_vector_reviews(tokenized_reviews, vocab_dict)
    return one_hot_vectors, reviews

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Embeddings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# loads the GoogleNews word embeddings
def load_word2vec_embeddings():
    print("loading GoogleNews word embeddings...")
    return KeyedVectors.load_word2vec_format(WORD_VECTORS_FILE, binary=True)

# creates a list of average word embedding vectors from a list of tokenized sentences
def sentence_embeddings(tokenized_sentences, vectors):
    '''
    adds an </s> token to each sentence vector to ensure no empty sentences

    converts each token in a sentence to a word embedding vector, if it's in
    the vocabulary

    averages the word vectors to create a sentence average vector
    '''
    v = vectors
    word_vectors = [ np.array([ v[w] for w in sentence+['</s>'] if w in v.vocab ]) \
        for sentence in tokenized_sentences ]
    sentence_average_vectors = [ np.mean(s, axis=0) for s in word_vectors ]
    return sentence_average_vectors

# loads GoogleNews embeddings, then computes embeddings for given tokens
def tokens_to_embeddings(tokenized_reviews):
    v = load_word2vec_embeddings()
    sentence_average_vectors = sentence_embeddings(tokenized_reviews, v)
    return sentence_average_vectors

# converts a file into a list of sentence avg embeddings, given a set of word embeddings
def test_file_to_embeddings(filepath, embeddings):
    tokenized_reviews, reviews = test_file_to_tokens(filepath)
    sentence_average_vectors = sentence_embeddings(tokenized_reviews, embeddings)
    return sentence_average_vectors, reviews

#~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenating POS Tags~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# pads the length of all the vectors in the list to the length of the longest vector
def pad_length_to_max(review_tag_vectors):
    max_len = max([len(review) for review in review_tag_vectors])
    tag_vector_len = ((max_len + 9) // 10) * 10
    review_tag_vectors = [ review + [-1] * (tag_vector_len - len(review)) \
                            for review in review_tag_vectors ]
    return review_tag_vectors

# converts a file into a list of tag indices for each review
def review_tag_indices(filepath):
    file_str = file_to_str(filepath)
    reviews = re.findall(r'^(.+)\|(.+)$', file_str, re.MULTILINE)
    tokenized_reviews = [ review.split() for review, rating in reviews ]
    print('pos tagging reviews...')
    tagged_reviews = [nltk.pos_tag(review) for review in tokenized_reviews]
    review_tag_vectors = [ [ TAGS_DICT.get(tag, 0) for word, tag in review ] \
                            for review in tagged_reviews ]
    return pad_length_to_max(review_tag_vectors)

# creates a single list of concatenated vectors from two given vector lists
def embedding_and_tag_vectors(embedding_vectors, tag_vectors):
    concatenated_vectors = []
    for i in range(len(embedding_vectors)):
        concatenated = np.concatenate([embedding_vectors[i], np.array(tag_vectors[i])])
        concatenated_vectors.append(concatenated)
    return concatenated_vectors
