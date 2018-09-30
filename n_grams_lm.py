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

pathGutenberg = r'D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg'
pathNewsData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/news_data'
pathImdbData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/imdb_data'

class N_Grams_LM:
    """
    N-gram language model.

    Initialize with N-grams from a collection of documents.
    Can generate random sentences.
    """

    def __init__(self):
        self.files = []                     # Files in corpus
        # ... over whole corpus ...
        self.sents = []                     # Sentences
        self.tokens = []                    # Word tokens
        self.infrequent = defaultdict(int)  # Infrequent tokens -> 'UNK'
        self.tokens_UNK = defaultdict(int)  # Tokens with 'UNK's
        self.grams = defaultdict(int)       # N-grams across corpus
        # ... by file in corpus ...
        self.sents_in_files = {}            # Sentences, by file
        self.tokens_in_files = {}           # Word tokens, by file
        self.infrequent_in_files = {}       # Infrequent tokens, by file
        self.tokens_UNK_in_files = {}       # Tokens with 'UNKs', by file
        self.grams_in_files = {}            # N-grams, by file

# ------------------------------------------------------------------------
# Preprocessing ---
# ------------------------------------------------------------------------

    def words_from_sents(self, sents):
        """
        Extract words from a list of sentence strings.
        Add '<s>' and '</s>' tokens, before and after each sentence.
        """
        words = []
        for sent in sents:
            words += [ '<s>' ]                  # start of sentence
            words += nltk.word_tokenize(sent)   # words in sentence
            words += [ '</s>' ]                 # end of sentence
        return words

    def words_from_sent_tokens(self, sents):
        """
        Extract words from a list of sentence token lists,
        as returned by nltk.corpus.gutenberg.sent_tokenize(text_string).
        Add '<s>' and '</s>' tokens, before and after each sentence.
        """
        words = []
        for words_in_sent in sents:
            words += [ '<s>' ]      # start of sentence
            words += words_in_sent  # words in sentence
            words += [ '</s>' ]     # end of sentence
        return words

    def get_infrequent_tokens(self, tokens, TOO_FEW):
        """
        Compile a list of infrequent tokens and their counts.
        An infrequent token is one that occurs TOO_FEW or fewer times.
        """
        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        return { token : counts[token] \
                       for token in counts if counts[token] <= TOO_FEW }

    def infrequent_to_UNK(self, tokens, infrequent):
        """
        Replace infrequent words with 'UNK'.
        """
        tokens_prepped = tokens.copy()
        for i, token in enumerate(tokens_prepped):
            if token in infrequent:
                tokens_prepped[i] = 'UNK'
        return tokens_prepped

    def preprocess_file_to_tokens(self, dirPath, fnx):
        """
        Preprocess a file ...
        1. remove blank lines
        2. change newline characters to blanks
        3. change multiple blanks to single blanks
        4. tokenize the file (nltk word tokenize)
        """
        fnxPath = os.path.join(dirPath, fnx)
        with open(fnxPath) as f:
            fnx_data = f.read()
        fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
        fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)

        fnx_sents = nltk.sent_tokenize(fnx_data_sb)
        fnx_words = self.words_from_sents(fnx_sents)
        
        return fnx_sents, fnx_words

    def preprocess_files(self, dirPath, TOO_FEW):
        """
        PHASE I -
        For each file in the corpus:
        1. Preprocess to word tokens, with <s> and </s> marking sentences'
           blank lines removed, newlines converted to blanks, and
           multiple blanks collapsed to single blanks.
        2. Collect infrequent tokens, to compare with tokens
           that are infrequent over the whole corpus.
        PHASE II -
        Once tokens from all of the files are available,
        compute the infrequent tokens over the whole corpus.
        PHASE III -
        For each file in the corpus:
        1. Replace the infrequent tokens with 'UNK'
        """
        self.files = os.listdir(dirPath)
        # Collect sentences and word tokens
        # per file and for the whole corpus.
        # Collect infrequent tokens per file
        # (mainly for collecting statistics).
        for fnx in files:
            fnxPath = os.path.join(dirPath, fnx)
            if fnx != 'README' and os.path.isfile(fnxPath):
                sents, tokens = self.preprocess_file_to_tokens(dirPath, fnx)
                infrequent = self.get_infrequent_tokens(tokens, TOO_FEW)
                self.sents_in_files[fnx] = sents
                self.tokens_in_files[fnx] = tokens
                self.infrequent_in_files[fnx] = infrequent
                self.sents += sents
                self.tokens += tokens
        # Compute infrequent tokens over the whole corpus
        self.infrequent = self.get_infrequent_tokens(self.tokens, TOO_FEW)
        self.tokens_UNK = self.infrequent_to_UNK(self.tokens, self.infrequent)
        # Replace the infrequent tokens with 'UNK'
        for fnx in files:
            fnxPath = os.path.join(dirPath, fnx)
            if fnx != 'README' and os.path.isfile(fnxPath):
                fnx_tokens = self.tokens_in_files[fnx]
                fnx_tokens_UNK = \
                    self.infrequent_to_UNK(fnx_tokens, self.infrequent)
                self.tokens_UNK_in_files[fnx] = fnx_tokens_UNK

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
            for i in range(len(tokens)-(n-1)):
                gramList = []
                for j in range(n):
                    gramList += [ tokens[i+j] ]
                gram = tuple(gramList)
                if gram not in self.grams:
                    countNewInAll += 1
                grams_in_doc[gram] += 1
                self.grams[gram] += 1
        return grams_in_doc, countNewInAll

    def set_n_grams_from_files(self, dirPath, n, TOO_FEW):
        """
        Process all the files in the given directory
        (except the README) to set the N-grams in the instance.
        For each file:
          Preprocess to tokens, with <s> and </s> delimiting sentences,
          and infrequent tokens replaced by 'UNK'.
          Add the N-grams and a count for each N-gram
          to a dictionary of N-grams for the file,
          to a dictionary of N-grams for the entire corpus.
        Report the counts of N-grams found in each file, and in total.
        """
        print("--------------------------------------------------")
        print("-- Initialize %d-gram model for the files in directory" % ( n ))
        print("--   %s" % ( dirPath ))
        print("-- with tokens that occur %d or fewer times" % ( TOO_FEW ))
        print("-- replaced by 'UNK'")
        print("--------------------------------------------------")
        self.preprocess_files(dirPath, TOO_FEW)

        print("----+- Files -+----+----+----| %d-Grams |-- New --|" % (n))
        for fnx in files:
            fnxPath = os.path.join(dirPath, fnx)
            if fnx != 'README' and os.path.isfile(fnxPath):
                fnx_tokens = self.tokens_in_files[fnx]
                fnx_grams, countNew = self.add_grams(n, fnx_tokens)
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
    TOO_FEW = 5

    print("-- test infrequent_to_UNK() --")
    files = os.listdir(pathGutenberg)
    fnx = files[0]
    with open(pathGutenberg + r'/' + fnx) as f:
        fnx_data = f.read()
    fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
    fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)
    fnx_sents = nltk.corpus.gutenberg.sents(fnx)
    fnx_sents_nltk = nltk.sent_tokenize(fnx_data_sb)
    fnx_tokens = model.words_from_sents(fnx_sents_nltk)
    fnx_tokens_nltk = nltk.word_tokenize(fnx_data_sb)
    fnx_unk = model.get_infrequent_tokens(fnx_tokens, TOO_FEW)
    fnx_tokens_prepped = model.infrequent_to_UNK(fnx_tokens, fnx_unk)
    print("Count UNK tokens:", len(fnx_unk))
    print("First 30 UNK tokens --")
    print(list(fnx_unk.items())[:30])
    print("Count prepped tokens:", len(fnx_tokens_prepped))
    print("First 30 prepped tokens --")
    print(fnx_tokens_prepped[:30])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("-- test preprocess_file_to_tokens() --")
    print("prep tokens for %s" % (fnx))
    fnx_0_sents, fnx_0_tokens = \
        model.preprocess_file_to_tokens(pathGutenberg, fnx)
    fnx_0_unk = model.get_infrequent_tokens(fnx_0_tokens, TOO_FEW)
    fnx_0_tokens_prepped = model.infrequent_to_UNK(fnx_0_tokens, fnx_0_unk)
    print("Count sentences:", len(fnx_0_sents))
    print("First 5 sentences --")
    print(fnx_0_sents[:5])
    print("Count of word tokens:", len(fnx_0_tokens))
    print("First 30 word tokens --")
    print(fnx_0_tokens[:30])
    print("Count UNK tokens:", len(fnx_0_unk))
    print("First 30 UNK tokens --")
    print(list(fnx_0_unk.items())[:30])
    print("Count of prepped tokens:", len(fnx_0_tokens_prepped))
    print("First 30 prepped tokens --")
    print(fnx_0_tokens_prepped[:30])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("-- test add_grams() --")
    fnx_0_4_grams, countNewInAll = model.add_grams(4, fnx_0_tokens)
    fnx_0_4_gram_count = len(fnx_0_4_grams)
    fnx_0_4_repeats = [ (gram, fnx_0_4_grams[gram],) \
        for gram in fnx_0_4_grams if fnx_0_4_grams[gram] > 1 ]
    fnx_0_4_repeats_total = sum([ count for gram, count in fnx_0_4_repeats ])
    fnx_0_4_grams_total = sum([count for gram, count in fnx_0_4_grams.items()])
    print("%-30s%10d%10d" % (fnx, fnx_0_4_gram_count, countNewInAll) )
    print("Count of 4-grams used more than once:", len(fnx_0_4_repeats))
    print("Total instances of repeated 4-grams:", fnx_0_4_repeats_total)
    print("Count of 4-gram instances in text:", fnx_0_4_grams_total)
    print("Sample 30 repeated 4-grams ---")
    print(fnx_0_4_repeats[:30])
    print("First 30 4-grams --")
    print(list(fnx_0_4_grams.items())[:30])
    print("Last 30 4-grams --")
    print(list(fnx_0_4_grams.items())[-30:])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    for n in range(1, 7):
        print("-- test set_n_grams_from_files()- %d-grams --" % (n))
        model = N_Grams_LM()
        model.set_n_grams_from_files(pathGutenberg, n, 5)
        grams = [ (gram, model.grams[gram],) for gram in model.grams]
        print("Sample first 30 %d-grams found --" % (n))
        print(grams[:30])
        print("Sample last 30 %d-grams found --" % (n))
        print(grams[-30:])
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

"""
if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    bar = "|" + 5 * "----+----|"
    print(bar)
    print("-- 3.2 Perplexity --")
    print(bar)

    print("-- building blocks --")
    pmodel = N_Grams_LM()
    pfiles = os.listdir(pathNewsData)
    pfnx = pfiles[0]
    pfnxPath = os.path.join(pathNewsData, pfnx)
    with open(pfnxPath) as f:
        pfnx_data = f.read()
    pfnx_data_nnl = re.sub(r'\n', ' ', pfnx_data)
    pfnx_data_sb = re.sub(r"( )+", ' ', pfnx_data_nnl)
    pfnx_sents = nltk.sent_tokenize(pfnx_data_sb)
    pfnx_words = nltk.word_tokenize(pfnx_data_sb)
    pfnx_word_tokens = pmodel.words_from_sents(pfnx_sents)
    pfnx_unk, pfnx_tokens_prepped = pmodel.infrequent_to_UNK(pfnx_word_tokens)
    print("Count UNK tokens:", len(pfnx_unk))
    print("First 30 UNK tokens --")
    print(pfnx_unk[:30])
    print("Count prepped tokens:", len(pfnx_tokens_prepped))
    print("First 30 prepped tokens --")
    print(pfnx_tokens_prepped[:30])
    

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
"""
