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

# ------------------------------------------------------------------------
# Constants ---
# ------------------------------------------------------------------------
TOK_SS = '<s>'  # start sentence
TAG_SS = '$S'
TOK_ES = '</s>' # end sentence
TAG_ES = 'S$'

pathToyPOS = r'D:/Documents/NLP/NEU_CS6120/assignment_1/toyPOS'
pathBrownData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/brown'
pathTestDataFile = r'D:/Documents/NLP/NEU_CS6120/science_sample.txt'


# ------------------------------------------------------------------------
# Main Class - HMM POS Bigram Model ---
# ------------------------------------------------------------------------
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
        if self.DEBUG:
            print("Random value, r:", r, ", Item list size:", len(cps))
        entry = None
        first = 0
        last = len(cps) - 1
        found = False
        while first < last:     # while interval size > 1
            i = (first + last) // 2
            entry = cps[i]
            prob = entry[1];
            if self.DEBUG and i < 20:
                print("---", first, i, last, ":", entry, prob)
            if r < prob:
                last = i        # in this or earlier interval
            else:
                first = i + 1   # in later interval

        return cps[last]

# ------------------------------------------------------------------------
# HMM Probabilities - transition and emission ---
# ------------------------------------------------------------------------

    def _emission_probabilities(self, count_tag_unigrams, count_word_tag_pairs):
        """
        Calculate emission probability (alpha-smoothed):
            P(w_i | t_i) = (C(w_i, t_i) + alpha) / (C(t_i) + alpha * V)
            where V = count unique word tag pairs and alpha = 0.1
        Inputs:
            count_word_tags:
                { ( w_i, t_i ) : count, ... }
        Outputs:
            emission probabilities:
                { ( w_i , t_i ) : probability, ... }
            word emission probabilities:
                { t_i : [ ( w_i , probability ), ... ], ... }
        """
        alpha = 0.1
        V = len(count_word_tag_pairs)   # count of unique word tag pairs
        alpha_V = alpha * V
        # Compute probability of unseen word tag pair (count = 0)
        emission_probabilities_unseen = defaultdict(lambda: 1.0 / V)
        for tag, tag_count in count_tag_unigrams.items():
            unseen_probability = alpha / (tag_count + alpha_V)
            emission_probabilities_unseen[tag] = unseen_probability
        # Calculate the emission probability P(w_i | t_i)
        emission_probabilities = { }
        word_emission_probabilities = defaultdict(list)
        for word_tag_pair, word_tag_count in count_word_tag_pairs.items():
            word, tag = word_tag_pair
            tag_count = count_tag_unigrams[tag]
            probability = (float(word_tag_count) + alpha) \
                        / (tag_count + alpha_V)
            emission_probabilities[word_tag_pair] = probability
            word_emission_probabilities[tag] += [ ( word, probability ) ]
        return emission_probabilities, word_emission_probabilities, \
               emission_probabilities_unseen

    def _transition_probabilities(self, count_tag_unigrams, count_tag_bigrams):
        """
        Calculate transition probability (alpha-smoothed):
            P(t_i-1, t_i) = (C(t_i-1, t_i) + alpha) / (C(t_i-1) + alpha * V)
            where V = count unique bigrams, alpha = 0.1
        Inputs:
            count_tag_bigrams:
                { ( t_i-1, t_i ) : count, ... }
        Outputs:
            transition probabilities:
                { ( t_i-1, t_i ) : probability, ... }
            tag transition probabilities:
                { t_i-1 : [ ( t_i, probability ) ... ], ... }
        """
        alpha = 0.1
        V = len(count_tag_bigrams)  # count of unique tag bigrams
        alpha_V = alpha * V
        # Compute probability of unseen bigram (count = 0)
        transition_probabilities_unseen = defaultdict(lambda: 1.0 / V)
        for tag, tag_count in count_tag_unigrams.items():
            unseen_probability = alpha / (tag_count + alpha_V)
            transition_probabilities_unseen[tag] = unseen_probability
        # Calculate the transition probability P(t_i-1, t_i)
        transition_probabilities = { }
        tag_transition_probabilities = defaultdict(list)
        for bigram, bigram_count in count_tag_bigrams.items():
            prev_tag, tag = bigram
            prev_tag_count = count_tag_unigrams[prev_tag]
            probability = (float(bigram_count) + alpha) \
                        / (prev_tag_count + alpha_V)
            transition_probabilities[bigram] = probability
            tag_transition_probabilities[prev_tag] += [ ( tag, probability, ) ]
        return transition_probabilities, tag_transition_probabilities, \
               transition_probabilities_unseen

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
        word_tag_pairs_UNK = []
        for word, tag in word_tag_pairs:
            if word in count_infrequent:
                word = 'UNK'
            word_tag_pairs_UNK += [ ( word, tag ) ]
        return count_words, count_infrequent, word_tag_pairs_UNK

    def _unknown_word_tags(self, word_tag_pairs, count_infrequent):
        """
        Return a copy of the word counts dictionary
        with the infrequent words replaced by a 'UNK' entry
        that has count the sum of their counts.
        Inputs:
            count_word_tags:   { ( word, tag ) : count, ... }
            count_infrequent:  { word : count, ... }
        Outputs:
            count_word_tag_pairs_UNK:  { ( word, tag  ): count, ...
                                    ( 'UNK', tag ) : count_unk, ... }
            ... where count_unk is the sum of the counts of the
                infrequent words with that tag.
        """
        count_word_tag_pairs = defaultdict(int)
        for word_tag in word_tag_pairs:
            word, tag = word_tag
            count_word_tag_pairs[word_tag] += 1
        count_word_tag_pairs_UNK = count_word_tag_pairs.copy()
        for word_tag, count in count_word_tag_pairs.items():
            word, tag = word_tag
            if word in count_infrequent:
                count_word_tag_pairs_UNK[('UNK', tag,)] += count
                del count_word_tag_pairs_UNK[word_tag]
        return count_word_tag_pairs_UNK

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
            word_tag_pairs += [ ( TOK_SS, TAG_SS, ) ]    # Start of sentence
            word_tag_pairs += pairs_in_sent             # words and tags
            word_tag_pairs += [ ( TOK_ES, TAG_ES, ) ]   # End of sentence
        return word_tag_pairs

    def _tagged_sentences_from_file(self, dirPath, fnx):
        fnxPath = os.path.join(dirPath, fnx)
        re_nl = re.compile(r'\n')
        re_sb = re.compile(r'( )+')
        sents_in_file = []
        with open(fnxPath) as f:
            print(fnx)
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
            sents += fnx_sents
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
        # identify infrequent words and replace with ('UNK',tag) counts
        self.count_words, self.count_infrequent, self.word_tag_pairs_UNK = \
            self._infrequent_words(self.word_tag_pairs, self.TOO_FEW)
        self.count_word_tag_pairs_UNK = \
            self._unknown_word_tags(self.word_tag_pairs, self.count_infrequent)
        # bigrams and counts, from word tag pairs with infrequent set to UNK
        self.count_word_tags, self.count_tag_unigrams, self.count_tag_bigrams = \
            self._counts_from_word_tag_pairs(self.word_tag_pairs_UNK)
        # transition and emission probabilities
        self.pTrans, self.pTagTrans, self.pTransUnseen = \
            self._transition_probabilities( \
                self.count_tag_unigrams, self.count_tag_bigrams)
        self.pEmiss, self.pTagEmiss, self.pEmissUnseen = \
            self._emission_probabilities( \
                self.count_tag_unigrams, self.count_word_tags)
        self.pEmUNK, self.pTagEmUNK, self.pEmUNKUnseen = \
            self._emission_probabilities( \
                self.count_tag_unigrams, self.count_word_tag_pairs_UNK)
        # cumulative probabilities for random choosing
        self.pCumTrans = self._cumulative_probabilities(self.pTagTrans)
        self.pCumEmiss = self._cumulative_probabilities(self.pTagEmiss)
        self.pCumEmUNK = self._cumulative_probabilities(self.pTagEmUNK)

    def reset(self):
        # ... over whole training set ...
        self.files = None                   # List of files in training set
        self.TOO_FEW = None                 # UNK if word count <= TOO_FEW
        self.sents = None                   # List of sentences
        self.tags = None                    # List of (word, tag) pairs
        # counts ...
        self.count_word_tags = None         # { (w_i, t_i) : count, .. }
        self.count_words = None             # { w_i : count, ... }
        self.count_infrequent = None        # { (w_i, t_i) : count, ... }
        self.count_word_tag_pairs_UNK = None     # { (w_i, t_i) : count, ... }
        self.count_tag_unigrams = None      # { (t_i) : count, ... }
        self.count_tag_bigrams = None       # { (t_i-1, t_i) : count, ... }
        # probabilities
        self.pTrans = None          # { (t_i-1, t_i) : P(t_i-1, t_i), ... }
        self.pEmiss = None          # { (w_i, t_i) : P(w_i | t_i), ... }
        self.pEmUNK = None          # { (w_i, t_i) : P(w_i | t_i), ... }
        # conditional probabilities
        self.pTagTrans = None       #  { t_i-1 : (t_i, P(t_i-1, t_i)), ... }
        self.pTagEmiss = None       #  { t_i   : (w_i, P(w_i  | t_i)), ... }
        self.pTagEmUNK = None       #  { t_i   : (w_i, P(w_i  | t_i)), ... }
        # cumulative conditional probabilities
        self.pCumTrans = None       # { t_i-1 : [ (t_i, cP(t_i-1, t_i)) ], ... }
        self.pCumEmiss = None       # { t_i   : [ (w_i, cP(w_i  | t_i)) ], ... }
        self.pCumEmUNK = None       # { t_i   : [ (w_i, cP(w_i  | t_i)) ], ... }

    def set_DEBUG(self, DEBUG=True):
        self.DEBUG=DEBUG

    def __init__(self, DEBUG=False):
        self.set_DEBUG(DEBUG)
        self.reset()

# ------------------------------------------------------------------------
# Sentence Generation ---
# ------------------------------------------------------------------------

    def _assemble_sentence(self, swt):
        sent = ""
        sent_tagged = ""
        ss = False
        for word, tag in swt:
            if tag == TAG_SS:
                ss = True
            elif tag != TAG_ES:
                if tag == 'np' or ss:
                    word = word.capitalize()
                    ss = False
                sent += word + ' '
                sent_tagged += word + "/"  + tag + ' '
            if tag == TAG_ES:
                sent = sent[:-1]
                sent_tagged = sent_tagged[:-1]
        return sent, sent_tagged

    def generate_sentence(self, pTrans, pEmiss, pCumTrans, pCumEmiss):
        swt = []    # sentence word/tag pairs
        stp = []    # sentence transition probabilities
        sep = []    # sentence emission probabilities
        # start of sentence word and tag
        word_tag = ( TOK_SS, TAG_SS, )
        swt += [ word_tag ]
        stp += [ 1.0 ]
        sep += [ 1.0 ]
        # Iterate choosing tags and words until end of sentence is chosen
        next_word = None
        next_tag = None
        while next_word != TOK_ES:
            # generate the next word/tag pair
            word, tag = word_tag
            tcps = pCumTrans[tag]   # List of cumulative transition probabilities
            next_tag_cumP = self._choose_by_probability(tcps)
            next_tag, tagCumP = next_tag_cumP
            ecps = pCumEmiss[next_tag]  # List of cumulative emission probabilities
            next_word_cumP = self._choose_by_probability(ecps)
            next_word, wordCumP = next_word_cumP
            # get the probabilities used
            tp = pTrans[( tag, next_tag, )]
            ep = pEmiss[( next_word, next_tag )]
            # record word/tag pair
            word_tag = ( next_word, next_tag )
            swt += [ word_tag ]
            stp += [ tp ]
            sep += [ ep ]
            # continue generating as long as the next word is not the end of sentence token
        sent, sent_tagged = self._assemble_sentence(swt)
        prob = np.prod(np.array(stp)) * np.prod(np.array(sep))
        return swt, stp, sep, sent, sent_tagged, prob

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
    fnx = files[-1]
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

    fnx_count_words, fnx_count_infrequent, fnx_word_tag_pairs_UNK = \
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

    fnx_count_word_tags, fnx_count_tag_unigrams, fnx_count_tag_bigrams = \
            hmm._counts_from_word_tag_pairs(fnx_word_tag_pairs_UNK)
    fnx_count_word_tag_pairs_UNK = \
        hmm._unknown_word_tags(fnx_word_tag_pairs, fnx_count_infrequent)
    fnx_count_word_tag_pairs_UNK_sum = sum([c for p, c in
                              fnx_count_word_tag_pairs_UNK.items()])
    print("Sum counts in count word tags UNK =", fnx_count_word_tag_pairs_UNK_sum)
    print("Length count word tags UNK:", len(fnx_count_word_tag_pairs_UNK))
    print("First 5 count word tags UNK:", list(fnx_count_word_tag_pairs_UNK.items())[:5])
    print("Last 5 count word tags UNK:", list(fnx_count_word_tag_pairs_UNK.items())[-5:])

    fnx_pTrans, fnx_pTagTrans, fnx_pTransUnseen = \
        hmm._transition_probabilities(fnx_count_tag_unigrams, fnx_count_tag_bigrams)
    print("Length transition probabilities:", len(fnx_pTrans))
    print("First 5 transition probabilities:", list(fnx_pTrans.items())[:5])
    print("Last 5 transition probabilities:", list(fnx_pTrans.items())[-5:])
    print("Length tag transition probabilities:", len(fnx_pTagTrans))
    print("First 5 tag transition probabilities:", list(fnx_pTagTrans.items())[:5])
    print("Last 5 tag transition probabilities:", list(fnx_pTagTrans.items())[-5:])

    fnx_pEmiss, fnx_pTagEmiss, fnx_pEmissUnseen = \
        hmm._emission_probabilities(fnx_count_tag_unigrams, fnx_count_word_tags)
    print("Length emission probabilities:", len(fnx_pEmiss))
    print("First 5 emission probabilities:", list(fnx_pEmiss.items())[:5])
    print("Last 5 emission probabilities:", list(fnx_pEmiss.items())[-5:])
    print("Length tag emission probabilities:", len(fnx_pTagEmiss))
    print("First 5 tag emission probabilities:", list(fnx_pTagEmiss.items())[:5])
    print("Last 5 tag emission probabilities:", list(fnx_pTagEmiss.items())[-5:])

    fnx_pEmUNK, fnx_pTagEmUNK, fnx_pEmUNKUnknown = \
        hmm._emission_probabilities(fnx_count_tag_unigrams, fnx_count_word_tag_pairs_UNK)
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

    print("Randomly generated characters ...")
    cps = [ ('a', 0.5), ('b', 0.6), ('c', 0.8), ('d', 0.95), ('e', 1.0) ]
    print(cps)
    sent = ""
    for i in range(30):
        char_prob = hmm._choose_by_probability(cps)
        char, prob = char_prob
        sent += char
        print(char_prob, end='')
    print()
    print(sent)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Randomly generated sentences ...")
    swp = stp = sep = sent = sent_tagged = prob = None
    for i in range(5):
        print("--- %d ---" % i)
        swt, stp, sep, sent, sent_tagged, prob = hmm.generate_sentence( \
            fnx_pTrans, fnx_pEmiss, fnx_pCumTrans, fnx_pCumEmiss)
        print("SWT---")
        print(swt)
        print("STP---")
        print(stp)
        print("SEP---")
        print(sep)
        print("SENTENCE ---")
        print(sent)
        print("TAGGED SENTENCE ---")
        print(sent_tagged)
        print("Sentence probability---")
        print(prob)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = pathToyPOS
    print("Test with all file in %s -----" % testPath)
    hmm.init(testPath, TOO_FEW=5)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Randomly generated sentences ...")
    swp = stp = sep = sent = sent_tagged = prob = None
    for i in range(5):
        print("--- %d ---" % i)
        swt, stp, sep, sent, sent_tagged, prob = hmm.generate_sentence( \
            fnx_pTrans, fnx_pEmiss, fnx_pCumTrans, fnx_pCumEmiss)
        print("SWT---")
        print(swt)
        print("STP---")
        print(stp)
        print("SEP---")
        print(sep)
        print("SENTENCE ---")
        print(sent)
        print("TAGGED SENTENCE ---")
        print(sent_tagged)
        print("Sentence probability---")
        print(prob)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = pathBrownData
    print("Test with all file in %s -----" % testPath)
    hmm.init(testPath, TOO_FEW=5)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Randomly generated sentences ...")
    swp = stp = sep = sent = sent_tagged = prob = None
    for i in range(5):
        print("--- %d ---" % i)
        swt, stp, sep, sent, sent_tagged, prob = hmm.generate_sentence( \
            fnx_pTrans, fnx_pEmiss, fnx_pCumTrans, fnx_pCumEmiss)
        print("SWT---")
        print(swt)
        print("STP---")
        print(stp)
        print("SEP---")
        print(sep)
        print("SENTENCE ---")
        print(sent)
        print("TAGGED SENTENCE ---")
        print(sent_tagged)
        print("Sentence probability---")
        print(prob)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
