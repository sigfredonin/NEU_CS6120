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

class neu_cs6120_a1:
    """
    Methods to solve CS6120 assignment 1 problems.
    """

    def write_prepped(self, dirPath, fnx):
        fn, ext = fnx.split(".")
        fn_prepped = fn + "_prepped." + ext
        preppedPath = os.path.join(dirPath, "prepped")
        if not os.path.exists(preppedPath):
            os.makedirs(preppedPath)
        with open(os.path.join(preppedPath, fn_prepped), 'w') as outfile:
            outfile.write(fnx_data_sb)
            outfile.close()

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
        """
        fnxPath = os.path.join(dirPath, fnx)
        with open(fnxPath) as f:
            fnx_data = f.read()
        fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
        fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)

        fnx_tokens = nltk.word_tokenize(fnx_data_sb)
        fnx_unk, fnx_tokens_prepped = self.infreq_to_UNK(fnx_tokens)
        
        return fnx_tokens_prepped

    def add_four_grams(self, tokens, four_grams_all):
        """
        Scan the tokens to collect the four-grams.
        For each four-gram found, increment its count
        in the four-grams count dictionary.
        Return
            the total number of four-grams found in the document
            the four-grams found in the document
            the total four-grams found new to the corpus
            the four-grams found in the corpus so far
        """
        countInDoc = 0
        countNewInAll = 0;
        four_grams_doc = defaultdict(int)
        if len(tokens) >= 4:
            for i in range(len(tokens)-3):
                gram = ( tokens[i], tokens[i+1], tokens[i+2], tokens[i+3], )
                if gram not in four_grams_doc:
                    countInDoc += 1
                four_grams_doc[gram] += 1
                if gram not in four_grams_all:
                    countNewInAll += 1
                four_grams_all[gram] += 1
        return countInDoc, four_grams_doc, countNewInAll, four_grams_all

    def process(self, dirPath, outPath):
        """
        Process all the files in the given directory
        (except the README).
        For each file:
          Preprocess to tokens, with infrequent tokens changed to 'UNK'
          Add the four-grams and a count for each four-gram to
          a dictionary of four-grams for the entire corpus.
        Return the list of four-grams and their counts.
        """
        total_grams = 0
        four_grams = defaultdict(int)
        files = os.listdir(dirPath)
        for fnx in files:
            fnxPath = os.path.join(dirPath, fnx)
            if fnx != 'README' and os.path.isfile(fnxPath):
                tokens = self.preprocess_file_to_tokens(dirPath, fnx)
                countInDoc, gramsInDoc, countNew, grams = \
                    self.add_four_grams(tokens, four_grams)
                total_grams += countNew
                print("%-30s%10d%10d" % (fnx, countInDoc, countNew) )
        print("Total 4-grams found:", total_grams)
        return total_grams, four_grams

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    solver = neu_cs6120_a1()
    files = os.listdir(pathGutenberg)
    fnx = files[0]
    with open(pathGutenberg + r'/' + fnx) as f:
        fnx_data = f.read()
    fnx_data_nnl = re.sub(r'\n', ' ', fnx_data)
    fnx_data_sb = re.sub(r"( )+", ' ', fnx_data_nnl)
    fnx_tokens = nltk.word_tokenize(fnx_data_sb)

    print("-- test infreq_to_UNK() --")
    fnx_unk, fnx_tokens_prepped = solver.infreq_to_UNK(fnx_tokens)
    print("Count UNK tokens:", len(fnx_unk))
    print("First 30 UNK tokens --")
    print(fnx_unk[:30])
    print("First 30 prepped tokens --")
    print(fnx_tokens_prepped[:30])

    print("-- test add_four_grams() --")
    fnx_0_tokens = solver.preprocess_file_to_tokens(pathGutenberg, fnx)
    gramsAll = defaultdict(int)
    fnx_0_4_gram_count, fnx_0_4_grams, countNewInAll, gramsAll = \
        solver.add_four_grams(fnx_0_tokens, gramsAll)
    fnx_0_repeats = [ (gram, fnx_0_4_grams[gram],) \
        for gram in fnx_0_4_grams if fnx_0_4_grams[gram] > 1 ]
    print("%-30s%10d%10d" % (fnx, fnx_0_4_gram_count, countNewInAll) )
    print("Count of 4-grams used more than once:", len(fnx_0_repeats))
    print("Sample 30 repeated 4-grams ---")
    print(fnx_0_repeats[:30])

    print("-- test process() --")
    print("----+ File ---+----+----+----| 4-Grams |-- New --|")
    gut_count, gut_grams = solver.process(pathGutenberg, pathGutenberg+r'/test')
    grams = [ (gram, gut_grams[gram],) for gram in gut_grams]
    print("--------------------------------------------------")
    print("Total 4-grams found:", gut_count)
    print("Sample first 30 4-grams found --")
    print(grams[:30])
    print("Sample last 30 4-grams found --")
    print(grams[-30:])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
