"""
NEU CS6120 Assignment 1
Problem 3 Language Modeling
3.1 Report 4-gram counts in the Gutenberg data extract.
3.2 Calculate average perplexity in news articles and imdb reviews.

Sig Nin
21 Sep 2018
"""
import nltk
import os
import re
from collections import defaultdict

pathGutenberg = r'D:\Documents\NLP\NEU_CS6120\assignment_1\gutenberg'
pathNewsData = r'D:\Documents\NLP\NEU_CS6120\assignment_1\news_data'
pathImdbData = r'D:\Documents\NLP\NEU_CS6120\assignment_1\imdb_data'

class N_Grams_LM:
    """
    N-gram language model.

    Initialize with N-grams from a collection of documents.
    Can generate random sentences.
    """

    def __init__(self):
        self.grams = defaultdict(int)   # N-grams across corpus
        self.tokens_in_files = {}       # Tokens, by file
        self.unknown_in_files = {}      # Unknown tokens, by file
        self.grams_in_files = {}        # N-grams, by file
        self.files = []                 # Files in corpus

# ------------------------------------------------------------------------
# Preprocessing ---
# ------------------------------------------------------------------------

    def infreq_to_UNK(self, tokens):
        """
        Replace infrequent words with 'UNK'.
        1. Count words occurrences.
        2. List words with fewer than 5 occurrences.
        3. Replace infrequent words with 'UNK'
        """
        TOO_FEW = 5
        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        fnx_infreq = [ ( token, counts[token], ) \
                       for token in counts if counts[token] <= TOO_FEW ]
        tokens_prepped = tokens.copy()
        for i in range(len(tokens)):
            token = tokens[i]
            if counts[token] <= TOO_FEW:
                tokens_prepped[i] = 'UNK'
            else:
                tokens_prepped[i] = token
        return fnx_infreq, tokens_prepped

    def preprocess_file_to_tokens(self, dirPath, fnx):
        """
        Preprocess a file ...
        1. remove blank lines
        2. change newline characters to blanks
        3. change multiple blanks to single blanks
        4. tokenize the file (nltk word tokenize)
        5. replace tokens with <= 5 instances with 'UNK'
        Return ...
            infrequent tokens
            tokens in file, infrequent replaced with 'UNK'
        """
        fnxPath = os.path.join(dirPath, fnx)
        with open(fnxPath) as f:
            fnx_data = f.read()
        fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
        fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)

        fnx_tokens = nltk.word_tokenize(fnx_data_sb)
        fnx_unk, fnx_tokens_prepped = self.infreq_to_UNK(fnx_tokens)
        
        return fnx_unk, fnx_tokens_prepped

# ------------------------------------------------------------------------
# N-grams ---
# ------------------------------------------------------------------------

    def add_grams(self, n, tokens):
        """
        Scan the tokens to collect the N-grams.
        Add each N-gram found to the document N-grams dictionary.
        Add new N-grams found to the corpus N-grams dictionary.
        Return
            the N-grams found in the document
            the total number of N-grams found new to the corpus
        """
        countNewInAll = 0;
        grams_in_doc = defaultdict(int)
        if len(tokens) >= n:
            for i in range(len(tokens)-n):
                gramList = []
                for j in range(n):
                    gramList += [ tokens[i+j] ]
                gram = tuple(gramList)
                if gram not in self.grams:
                    countNewInAll += 1
                grams_in_doc[gram] += 1
                self.grams[gram] += 1
        return grams_in_doc, countNewInAll

    def set_n_grams_from_files(self, n, dirPath, outPath):
        """
        Process all the files in the given directory
        (except the README) to set the N-grams in the instance.
        For each file:
          Preprocess to tokens, with infrequent tokens changed to 'UNK'
          Add the tokens to a dictionary of tokens for the file,
          and the tokens replaced with 'UNK' to a dictionary
          of unknown tokens for the file.
          Add the N-grams and a count for each N-gram
          to a dictionary of N-grams for the file,
          to a dictionary of N-grams for the entire corpus.
        Report the counts of N-grams found in each file, and in total.
        """
        print("--------------------------------------------------")
        print("-- Initialize %d-gram model for the files in" % ( n ))
        print("-- directory %s" % ( dirPath ))
        print("--------------------------------------------------")
        self.grams = defaultdict(int)
        self.tokens_in_files = {}
        self.unknown_in_files = {}
        self.grams_in_files = {}
        print("----+- Files -+----+----+----| %d-Grams |-- New --|" % (n))
        self.files = os.listdir(dirPath)
        for fnx in files:
            fnxPath = os.path.join(dirPath, fnx)
            if fnx != 'README' and os.path.isfile(fnxPath):
                unk, tokens = self.preprocess_file_to_tokens(dirPath, fnx)
                self.tokens_in_files[fnx] = tokens
                self.unknown_in_files[fnx] = unk
                fnx_grams, countNew = self.add_grams(n, tokens)
                self.grams_in_files[fnx] = fnx_grams
                print("%-30s%10d%10d" % (fnx, len(fnx_grams), countNew) )
        print("--------------------------------------------------")
        print("Total %d-grams found: %d" % (n, len(self.grams)))
        print("--------------------------------------------------")

# ------------------------------------------------------------------------
# Unsmoothed N-gram Probabilities ---
# ------------------------------------------------------------------------

    def unsmoothed_unigrams(self, words):
        """
        Compute unsmoothed unigram probabilities
        from a list of words.
        """
        dist_words = nltk.FreqDist(words)    # unigram counts
        pgrams = {}
        for gram in set(words):
            count_word = dist_words[gram]
            pgram = count_word / len(words)
            pgrams[gram] = pgram
        return pgrams

    def unsmoothed_ngrams(self, words, grams):
        """
        Compute unsmoothed N-gram probabilities
        from a list of words, and a list of N-grams.
        """
        dist_words = nltk.FreqDist(words)    # unigram counts
        dist_grams = nltk.FreqDist(grams)    # N-gram counts
        pgrams = {}
        for iGram in range(len(grams)):
            gram = grams[iGram]
            count_gram = dist_grams[gram]
            count_word_0 = dist_words[gram[0]]
            pgram = count_gram / count_word_0
            pgrams[gram] = pgram
        return pgrams

# ------------------------------------------------------------------------
# Alpha-smoothed N-gram Probabilities ---
# ------------------------------------------------------------------------

    def alpha_smoothed_unigrams(self, alpha, words):
        """
        Compute alpha-smoothed unigram probabilities
        from a list of words.
        """
        dist_words = nltk.FreqDist(words)    # unigram counts
        pgrams = {}
        for gram in set(words):
            count_word = dist_words[gram]
            pgram = (count_word + alpha) / (len(words) + (alpha * len(words)))
            pgrams[gram] = pgram
        return pgrams

    def alpha_smoothed_ngrams(self, alpha, words, grams):
        """
        Compute alpha-smoothed N-gram probabilities
        from a list of words, and a list of N-grams.
        """
        dist_words = nltk.FreqDist(words)    # unigram counts
        dist_grams = nltk.FreqDist(grams)    # N-gram counts
        pgrams = {}
        for iGram in range(len(grams)):
            gram = grams[iGram]
            count_gram = dist_grams[gram]
            count_word_0 = dist_words[gram[0]]
            pgram = (count_gram + alpha) / (count_word_0 + (alpha * len(words)))
            pgrams[gram] = pgram
        return pgrams

# ------------------------------------------------------------------------
# Cumulative Probabilities and Random Choosing ---
# ------------------------------------------------------------------------

    def cummulative_probabilities(self, utps):
        """
        Calculate cummulative probabilities for
        a list of unsmoothed Ngram probabilities.
        """
        utcps = []                                      # .. cummulative
        cummulative_probability = 0.0
        for n_gram in utps:
            cummulative_probability += utps[n_gram]
            utcps += [(n_gram, cummulative_probability, )]
        return utcps

    def choose_by_probability(self, utcps):
        """
        Choose an Ngram at random from a list of
          (Ngram, cummulative probability)
        so that each ngram has its own probability of being chosen.
        Use binary search.
        """
        from random import uniform
        cummulative_probability = utcps[-1][1]
        r = uniform(0.0, cummulative_probability)
        print("Random value, r:", r, ", Trigram list size:", len(utcps))
        entry = None
        first = 0
        last = len(utcps) - 1
        found = False
        while first < last:     # while interval size > 1
            i = (first + last) // 2
            entry = utcps[i]
            prob = entry[1];
            if i < 20:
                print("---", first, i, last, ":", entry, prob)
            if r < prob:
                last = i        # in this or earlier interval
            else:
                first = i + 1   # in later interval
            
        return utcps[last]

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = N_Grams_LM()

    print("-- test infreq_to_UNK() --")
    files = os.listdir(pathGutenberg)
    fnx = files[0]
    with open(pathGutenberg + r'/' + fnx) as f:
        fnx_data = f.read()
    fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
    fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)
    fnx_tokens = nltk.word_tokenize(fnx_data_sb)
    fnx_unk, fnx_tokens_prepped = model.infreq_to_UNK(fnx_tokens)
    print("Count UNK tokens:", len(fnx_unk))
    print("First 30 UNK tokens --")
    print(fnx_unk[:30])
    print("Count prepped tokens:", len(fnx_tokens_prepped))
    print("First 30 prepped tokens --")
    print(fnx_tokens_prepped[:30])

    print("-- test preprocess_file_to_tokens() --")
    print("prep tokens for %s" % (fnx))
    fnx_0_unk, fnx_0_tokens = \
        model.preprocess_file_to_tokens(pathGutenberg, fnx)
    print("Count UNK tokens:", len(fnx_0_unk))
    print("First 30 UNK tokens --")
    print(fnx_0_unk[:30])
    print("Count of prepped tokens:", len(fnx_0_tokens))
    print("First 30 prepped tokens --")
    print(fnx_0_tokens[:30])

    print("-- test add_grams() --")
    fnx_0_4_grams, countNewInAll = model.add_grams(4, fnx_0_tokens)
    fnx_0_4_gram_count = len(fnx_0_4_grams)
    fnx_0_repeats = [ (gram, fnx_0_4_grams[gram],) \
        for gram in fnx_0_4_grams if fnx_0_4_grams[gram] > 1 ]
    print("%-30s%10d%10d" % (fnx, fnx_0_4_gram_count, countNewInAll) )
    print("Count of 4-grams used more than once:", len(fnx_0_repeats))
    print("Sample 30 repeated 4-grams ---")
    print(fnx_0_repeats[:30])

    for n in range(1, 7):
        print("-- test set_n_grams_from_files()- %d-grams --" % (n))
        model = N_Grams_LM()
        model.set_n_grams_from_files(n, pathGutenberg, pathGutenberg+r'/test')
        grams = [ (gram, model.grams[gram],) for gram in model.grams]
        print("Sample first 30 %d-grams found --" % (n))
        print(grams[:30])
        print("Sample last 30 %d-grams found --" % (n))
        print(grams[-30:])
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")
