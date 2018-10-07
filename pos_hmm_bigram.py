"""
NEU CS6120 Assignment 1
Problem 4 POS Tagging - Hidden Markov Model

The training set is a collection of files from the Brown corpus.
The training set files have sentences of tokenized tagged words,
    w_1/t_1 w_2/t_2 w_3/t_3 ... w_k-1/t_k-1 w_k/t_k 
one sentence per line, with leading white space.
Some lines are empty (i.e., just a newline).

4.1 Obtain frequency counts from all the training files counted together --
      C(w_i,t_i):     word-tag counts,
      C(t_i):         tag unigram counts,
      C(t_i-1, t_i):  tag bigram counts.
    Have to separate words/tags for this counting,
    and have to add beginning and end of sentence token-tag pairs,
        <s>/<$s> and </s>/<s$>.
    Identify infrequent words and replace with 'UNK' before counting.
4.2 Calculate transition probability:
      P(t_i-1, t_i) = C(t_i-1, t_i) / C(t_i-1)
4.3 Calculate emission probability:
      P(w_i | t_i) = C(w_i, t_i) / C(t_i)
4.4 Generate 5 random sentences using HMM.
    Output each sentence (with its POS tags),
    and the probability of it being generated.
    This uses the probabilities of the whole training vocabulary,
    including infrequent words.
4.5 Use the Viterbi algorithm (NLP ed3 Fig. 8.5) to
    derive the most probable tag sequence for each word
    in the test dataset:
        <sentence = ID=1>
        word, tag
        word, tag
        ...
        word, tag
        <EOS>
        <sentence = ID=1>
        word, tag
        word, tag
        ...
        word, tag
        <EOS>
        ...
    The test data set contains word tokens in this format, but without tags.
    This uses the probabilities of the words-tag pairs where infrequent words
    are collapsed into 'UNK'-tag pairs.

Sig Nin
03 Oct 2018
"""
import nltk
import numpy as np
import os
import re
from collections import defaultdict

pathToyPOS = r'D:/Documents/NLP/NEU_CS6120/assignment_1/toyPOS'
pathBrownData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/brown'
pathTestDataFile = r'D:/Documents/NLP/NEU_CS6120/science_sample.txt'

class POS_HMM_BiGram:
    """
    Bigram HMM POS model.

    Initialize with counts from a collection of documents.
    Can generate random sentences.
    Can generate sequences of likely tags for words in sentences,
    using the Viterbi algorithm.
    """

# ------------------------------------------------------------------------
# Cumulative Probabilities and Random Choosing ---
# ------------------------------------------------------------------------

    def _cumulative_probabilities_for_prior(self, probabilities):
        """
        Calculate cumulative probabilities for a list of probabilities.
        Input:
            List of probabilities for each possible successor:
            [ ( successor, probability ), ... ]
        Output:
            List of cumulative probabilities for each possible successor:
            [ ( successor, cumulative probability ), ... ]
        """
        cps = []                                      # .. cumulative
        cumulative_probability = 0.0
        for s, p in probabilities:
            cumulative_probability += p
            cps += [(s, cumulative_probability, )]
        return cps

    def _cumulative_probabilities(self, successor_probabilities):
        """
        Calculate cumulative probabilities for a list of succesor probabilities.
        The successor probabilities for each prior sum to 1.
        The cumulative successor probabilities end in 1.
        Input:
            List of probabilities for each possible successor for each prior:
            { prior : [ ( successor, probability ), ... ] ), ... }
        Output:
            List of cumulative probabilities for each possible successor for each prior:
            { prior, [ ( successor, cumulative probability ), ... ] ), ... }
        """
        scps = { }
        for prior, probabilities in successor_probabilities.items():
            cps = self._cumulative_probabilities_for_prior(probabilities)
            last, cp = cps[-1]
            if abs(1.0 - cp) > 1e-14:
                print("Warning: Probabilities don't add to 1.0", prior, last, cp)
            cps[-1] = ( last, 1.0 )
            scps[prior] = cps
        return scps

    def _choose_by_probability(self, cps):
        """
        Choose an item at random from a list of
          (item, cumulative probability)
        so that each item has its own probability of being chosen.
        Use binary search.
        """
        from random import uniform
        cumulative_probability = cps[-1][1]
        r = uniform(0.0, cumulative_probability)
        print("Random value, r:", r, ", Item list size:", len(cps))
        entry = None
        first = 0
        last = len(cps) - 1
        found = False
        while first < last:     # while interval size > 1
            i = (first + last) // 2
            entry = cps[i]
            prob = entry[1];
            if i < 20:
                print("---", first, i, last, ":", entry, prob)
            if r < prob:
                last = i        # in this or earlier interval
            else:
                first = i + 1   # in later interval
            
        return cps[last]

# ------------------------------------------------------------------------
# HMM Probabilities - transition and emission ---
# ------------------------------------------------------------------------

    def _emission_probabilities(self, count_word_tags):
        """
        Calculate emission probability:
            P(w_i | t_i) = C(w_i, t_i) / C(t_i)
        Inputs:
            count_word_tags:
                { ( ( w_i, t_i ) : count )... }
        Outputs:
            emission probabilities:
                { { ( w_i , t_i ) : probability ), ... }
        """
        # Extract tag counts from the word/tag pair counts
        tag_counts = defaultdict(int)
        for word_tag_pair, word_tag_count in count_word_tags.items():
            word, tag = word_tag_pair
            tag_counts[tag] += word_tag_count
        # Calculate the emission probability P(w_i | t_i)
        emission_probabilities = { }
        word_emission_probabilities = defaultdict(list)
        for word_tag_pair, word_tag_count in count_word_tags.items():
            word, tag = word_tag_pair
            tag_count = tag_counts[tag]
            probability = float(word_tag_count) / tag_count
            emission_probabilities[word_tag_pair] = probability
            word_emission_probabilities[tag] += [ ( word, probability ) ]
        return emission_probabilities, word_emission_probabilities

    def _transition_probabilities(self, count_tag_bigrams):
        """
        Calculate transition probability:
            P(t_i-1, t_i) = C(t_i-1, t_i) / C(t_i-1)
        Inputs:
            count_tag_bigrams:
                { { ( t_i-1, t_i ) : count ), ... }
        Outputs:
            transition probabilities:
                { { ( t_i-1, t_i ) : probability ), ... }
            tag transition probabilities:
                { { t_i-1 : [ ( t_i, probability ) ... ] }, ... }
        """
        # Extract prev_tag counts from the prev_tag/tag pair counts
        tag_counts = defaultdict(int)
        for bigram, bigram_count in count_tag_bigrams.items():
            prev_tag, tag = bigram
            tag_counts[prev_tag] += bigram_count
        # Calculate the transition probability P(t_i-1, t_i)
        transition_probabilities = { }
        tag_transition_probabilities = defaultdict(list)
        for bigram, bigram_count in count_tag_bigrams.items():
            prev_tag, tag = bigram
            prev_tag_count = tag_counts[prev_tag]
            probability = float(bigram_count) / prev_tag_count
            transition_probabilities[bigram] = probability
            tag_transition_probabilities[prev_tag] += [ ( tag, probability, ) ]
        return transition_probabilities, tag_transition_probabilities

# ------------------------------------------------------------------------
# Infrequent and unknown words, conversion to 'UNK' ---
# ------------------------------------------------------------------------

    def _infrequent_words(self, word_tag_pairs, TOO_FEW):
        """
        Return the word counts and infrequent word counts
        in dictionaries with entries (word, tag) : count.
        Inputs:
            word_tag_pairs: [ (word, tag), ... ]
            TOO_FEW: word is infrequent if count <= TOO_FEW
        Outputs:
            count_words:       { word : count, ... }
            count_infrequent:  { ( word ) : count, ... }
        """
        count_words = defaultdict(int)
        for word, tag in word_tag_pairs:
            count_words[word] += 1
        count_infrequent = defaultdict(int)
        for word, count in count_words.items():
            if count <= TOO_FEW:
                count_infrequent[word] += count
        return count_words, count_infrequent

    def _unknown_word_tags(self, count_word_tags, count_infrequent):
        """
        Return a copy of the word counts dictionary
        with the infrequent words replaced by a 'UNK' entry
        that has count the sum of their counts.
        Inputs:
            count_word_tags:   { ( word, tag ) : count, ... }
            count_infrequent:  { word : count, ... }
        Outputs:
            count_word_tags_UNK:  { ( word, tag  ): count, ...
                                    ( 'UNK', tag ) : count_unk, ... }
            ... where count_unk is the sum of the counts of the
                infrequent words with that tag.
        """
        count_word_tags_UNK = count_word_tags.copy()
        for word_tag, count in count_word_tags.items():
            word, tag = word_tag
            if word in count_infrequent:
                count_word_tags_UNK[('UNK', tag,)] += count
                del count_word_tags_UNK[word_tag]
        return count_word_tags_UNK

# ------------------------------------------------------------------------
# Sentences, words, tags and counts ---
# ------------------------------------------------------------------------

    def _counts_from_word_tag_pairs(self, word_tag_pairs):
        count_word_tags    = defaultdict(int)
        count_tag_unigrams = defaultdict(int)
        count_tag_bigrams  = defaultdict(int)
        tag_prev = None
        for pair in word_tag_pairs:
            word, tag = pair
            count_word_tags[pair] += 1
            tag_unigram = ( tag, )
            count_tag_unigrams[tag_unigram] += 1
            if tag_prev != None:
                tag_bigram = ( tag_prev, tag, )
                count_tag_bigrams[tag_bigram] += 1
            tag_prev = tag
        return count_word_tags, count_tag_unigrams, count_tag_bigrams

    def _tags_from_sentences(self, sents):
        p = re.compile(r'(\S+)/(\S+)')
        word_tag_pairs = []
        for sent in sents:
            pairs_in_sent = [ (word.lower(), tag) for word, tag in p.findall(sent) ]
            word_tag_pairs += [ ( '<s>', '<$s>', ) ]    # Start of sentence
            word_tag_pairs += pairs_in_sent             # words and tags
            word_tag_pairs += [ ( '</s>', '<s$>', ) ]   # End of sentence
        return word_tag_pairs

    def _tagged_sentences_from_file(self, dirPath, fnx):
        fnxPath = os.path.join(dirPath, fnx)
        re_nl = re.compile(r'\n')
        re_sb = re.compile(r'( )+')
        sents_in_file = []
        with open(fnxPath) as f:
            for line in f:
                nnl = re_nl.sub(' ', line)      # '\n' -> ' '
                sb  = re_sb.sub(' ', nnl)       # ' '+ -> ' '
                if sb != ' ':
                    sents_in_file += [ sb ]
        return sents_in_file

    def _tagged_sentences_from_files(self, dirPath, files):
        sents = []
        for fnx in files:
            fnx_sents = self._tagged_sentences_from_file(dirPath, fnx)
            sents += [ fnx_sents ]
        return sents

# ------------------------------------------------------------------------
# Class constructor and training ---
# ------------------------------------------------------------------------

    def init(self, dirPath, TOO_FEW=5):
        self.files = os.listdir(dirPath)
        self.TOO_FEW = TOO_FEW
        # sentences, word/tag pairs, counts
        self.sents = self._tagged_sentences_from_files(dirPath, self.files)
        self.word_tag_pairs = self._tags_from_sentences(self.sents)
        self.count_word_tags, self.count_tag_unigrams, self.count_tag_bigrams = \
            self._counts_from_word_tag_pairs(self.word_tag_pairs)
        # identify infrequent words and replace with ('UNK',tag) counts
        self.count_words, self.count_infrequent = \
            self._infrequent_words(self.word_tag_pairs, self.TOO_FEW)
        self.count_word_tags_UNK = \
            self._unknown_word_tags(self.count_word_tags, self.count_infrequent)
        # transition and emission probabilities
        self.pTrans, self.pTagTrans = self._transition_probabilities( \
            self.count_tag_bigrams)
        self.pEmiss, self.pTagEmiss = self._emission_probabilities( \
            self.count_word_tags)
        self.pEmUNK, self.pTagEmUNK = self._emission_probabilities( \
            self.count_word_tags_UNK)
        # cumulative probabilities for random choosing
        self.pCumTrans = self._cumulative_probabilities(self.pTagTrans)
        self.pCumEmiss = self._cumulative_probabilities(self.pTagEmiss)
        self.pCumEmUNK = self._cumulative_probabilities(self.pTagEmUNK)

    def reset(self):
        # ... over whole training set ...
        self.files = None                   # List of files in training set
        self.TOO_FEW = None                 # UNK if word count <= TOO_FEW
        self.sents = None                   # List of sentences
        self.tags = None                    # (word, tag)
        self.count_word_tags = None         # (w_i, t_i) : count
        self.count_words = None             #  w_i : count
        self.count_infrequent = None        # (w_i, t_i) : count
        self.count_word_tags_UNK = None     # (w_i, t_i) : count
        self.count_tag_unigrams = None      # (t_i) : count
        self.count_tag_bigrams = None       # (t_i-1, t_i) : count
        self.pTrans = None                  # (t_i-1, t_i) : P(t_i-1, t_i)
        self.pEmiss = None                  # (w_i, t_i) : P(w_i | t_i)
        self.pEmUNK = None                  # (w_i, t_i) : P(w_i | t_i)
        self.pTagTrans = None               #  t_i-1 : (t_i, P(t_i-1, t_i))
        self.pTagEmiss = None               #  w_i   : (w_i, P(w_i  | t_i))
        self.pTagEmUNK = None               #  w_i   : (w_i, P(w_i  | t_i))
        self.pCumTrans = None               #  t_i-1 : [ (t_i, cP(t_i-1, t_i)) ]
        self.pCumEmiss = None               #  w_i   : [ (w_i, cP(w_i  | t_i)) ]
        self.pCumEmUNK = None               #  w_i   : [ (w_i, cP(w_i  | t_i)) ]

    def __init__(self):
        self.reset()

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = pathToyPOS
    hmm = POS_HMM_BiGram()
    TOO_FEW = 1

    files = os.listdir(testPath)
    fnx = files[0]
    print("--- ", fnx, " ---")

    fnx_sents = hmm._tagged_sentences_from_file(testPath, fnx)
    print("Len sentences:", len(fnx_sents))
    print("First 5 sentences:", fnx_sents[:5])
    print("last 5 sentences:", fnx_sents[-5:])

    fnx_word_tag_pairs = hmm._tags_from_sentences(fnx_sents)
    print("Len word tag pairs:", len(fnx_word_tag_pairs))
    print("First 5 word tag pairs:", fnx_word_tag_pairs[:5])
    print("Last 5 word tag pairs:", fnx_word_tag_pairs[-5:])

    fnx_count_word_tags, fnx_count_tag_unigrams, fnx_count_tag_bigrams = \
            hmm._counts_from_word_tag_pairs(fnx_word_tag_pairs)
    fnx_count_word_tags_sum = sum([c for p, c in
                                   fnx_count_word_tags.items()])
    fnx_count_tag_unigrams_sum = sum([c for p, c in
                                      fnx_count_tag_unigrams.items()])
    fnx_count_tag_bigrams_sum = sum([c for p, c in
                                     fnx_count_tag_bigrams.items()])
    print("Sum counts in count word tag pairs =", fnx_count_word_tags_sum)
    print("Length count word tag pairs:", len(fnx_count_word_tags))
    print("First 5 count word tag pairs:", list(fnx_count_word_tags.items())[:5])
    print("Last 5 count word tag pairs:", list(fnx_count_word_tags.items())[-5:])
    print("Length count tag unigrams:", len(fnx_count_tag_unigrams))
    print("Sum counts in count tag unigrams =", fnx_count_tag_unigrams_sum)
    print("First 5 count tag unigrams:", list(fnx_count_tag_unigrams.items())[:5])
    print("Last 5 count tag unigrams:", list(fnx_count_tag_unigrams.items())[-5:])
    print("Sum counts in count tag bigrams =", fnx_count_tag_bigrams_sum)
    print("Length count tag bigrams:", len(fnx_count_tag_bigrams))
    print("First 5 count tag bigrams:", list(fnx_count_tag_bigrams.items())[:5])
    print("Last 5 count tag bigrams:", list(fnx_count_tag_bigrams.items())[-5:])

    fnx_count_words, fnx_count_infrequent = \
            hmm._infrequent_words(fnx_word_tag_pairs, TOO_FEW)
    fnx_count_words_sum = sum([c for p, c in
                              fnx_count_words.items()])
    fnx_count_infrequent_sum = sum([c for p, c in
                              fnx_count_infrequent.items()])
    print("Sum counts in count words =", fnx_count_words_sum)
    print("Length count words:", len(fnx_count_words))
    print("First 5 count words:", list(fnx_count_words.items())[:5])
    print("Last 5 count words:", list(fnx_count_words.items())[-5:])
    print("Sum counts in count infrequent words =", fnx_count_infrequent_sum)
    print("Length count infrequent words:", len(fnx_count_infrequent))
    print("First 5 count infrequent words:", list(fnx_count_infrequent.items())[:5])
    print("Last 5 count infrequent words:", list(fnx_count_infrequent.items())[-5:])

    fnx_count_word_tags_UNK = \
        hmm._unknown_word_tags(fnx_count_word_tags, fnx_count_infrequent)
    fnx_count_word_tags_UNK_sum = sum([c for p, c in
                              fnx_count_word_tags_UNK.items()])
    print("Sum counts in count word tags UNK =", fnx_count_word_tags_UNK_sum)
    print("Length count word tags UNK:", len(fnx_count_word_tags_UNK))
    print("First 5 count word tags UNK:", list(fnx_count_word_tags_UNK.items())[:5])
    print("Last 5 count word tags UNK:", list(fnx_count_word_tags_UNK.items())[-5:])

    fnx_pTrans, fnx_pTagTrans = hmm._transition_probabilities( \
        fnx_count_tag_bigrams)
    print("Length transition probabilities:", len(fnx_pTrans))
    print("First 5 transition probabilities:", list(fnx_pTrans.items())[:5])
    print("Last 5 transition probabilities:", list(fnx_pTrans.items())[-5:])
    print("Length tag transition probabilities:", len(fnx_pTagTrans))
    print("First 5 tag transition probabilities:", list(fnx_pTagTrans.items())[:5])
    print("Last 5 tag transition probabilities:", list(fnx_pTagTrans.items())[-5:])

    fnx_pEmiss, fnx_pTagEmiss = hmm._emission_probabilities( \
        fnx_count_word_tags)
    print("Length emission probabilities:", len(fnx_pEmiss))
    print("First 5 emission probabilities:", list(fnx_pEmiss.items())[:5])
    print("Last 5 emission probabilities:", list(fnx_pEmiss.items())[-5:])
    print("Length tag emission probabilities:", len(fnx_pTagEmiss))
    print("First 5 tag emission probabilities:", list(fnx_pTagEmiss.items())[:5])
    print("Last 5 tag emission probabilities:", list(fnx_pTagEmiss.items())[-5:])

    fnx_pEmUNK, fnx_pTagEmUNK = hmm._emission_probabilities( \
        fnx_count_word_tags_UNK)
    print("Length emission probabilities UNK:", len(fnx_pEmUNK))
    print("First 5 emission probabilities UNK:", list(fnx_pEmUNK.items())[:5])
    print("Last 5 emission probabilities UNK:", list(fnx_pEmUNK.items())[-5:])
    print("Length tag emission probabilities UNK:", len(fnx_pTagEmUNK))
    print("First 5 tag emission probabilities UNK:", list(fnx_pTagEmUNK.items())[:5])
    print("Last 5 tag emission probabilities UNK:", list(fnx_pTagEmUNK.items())[-5:])

    fnx_pCumTrans = hmm._cumulative_probabilities(fnx_pTagTrans)
    print("Length cumulative tag transition probabilities UNK:", len(fnx_pCumTrans))
    print("First 5 cumulative tag transition probabilities UNK:", list(fnx_pCumTrans.items())[:5])
    print("Last 5 cumulative tag transition probabilities UNK:", list(fnx_pCumTrans.items())[-5:])

    fnx_pCumEmiss = hmm._cumulative_probabilities(fnx_pTagEmiss)
    print("Length cumulative tag emission probabilities:", len(fnx_pCumEmiss))
    print("First 5 cumulative tag emission probabilities:", list(fnx_pCumEmiss.items())[:5])
    print("Last 5 cumulative tag emission probabilities:", list(fnx_pCumEmiss.items())[-5:])

    fnx_pCumEmUNK = hmm._cumulative_probabilities(fnx_pTagEmUNK)
    print("Length cumulative tag emission probabilities UNK:", len(fnx_pCumEmUNK))
    print("First 5 cumulative tag emission probabilities UNK:", list(fnx_pCumEmUNK.items())[:5])
    print("Last 5 cumulative tag emission probabilities UNK:", list(fnx_pCumEmUNK.items())[-5:])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
