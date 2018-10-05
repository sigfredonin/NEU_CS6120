import re
import os
import pickle

from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# class Preprocessor:
#     '''
#     Contains methods for preprocessing files and strings to be used in NLP tasks.
#     '''

THRESHOLD = 5

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(fileDir)
input_path = os.path.join(parentDir, "resources/raw")
output_path = os.path.join(parentDir, "resources/processed")

# removes blank lines, replaces \n with space, removes duplicate spaces
def process_whitespace(str):
    no_new_line = re.sub(r'\n', " ", str)
    no_dup_spaces = re.sub(r'  +', " ", no_new_line)
    return no_dup_spaces

# tokenizes the given string and converts words to lowercase
def tokenize(processed_str):
    return list(map(lambda word: word.lower(), word_tokenize(processed_str)))

# replaces words low frequency words with UNK token
def repl_infreq_w_unknown(tokens):
    fdist = FreqDist(tokens)
    unknowns = list(filter(lambda word: fdist[word] < THRESHOLD, fdist))
    return list(map(lambda word: 'UNK' if word in unknowns else word, tokens))

# all in one in case I decide to delete the individual methods
def process_str(str):
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    # fix whitespace
    print("len str:", len(str))
    print("sample str:", str[:50])
    no_new_line = re.sub(r'\n', " ", str)
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print("len no_new_line:", len(no_new_line))
    print("sample no_new_line:", no_new_line[:50])
    no_dup_spaces = re.sub(r'  +', " ", no_new_line)
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print("len no_dup_spaces:", len(no_dup_spaces))
    print("sample no_dup_spaces:", no_dup_spaces[:50])

    # lowercase and tokenize
    tokens = list(map(lambda word: word.lower(), word_tokenize(no_dup_spaces)))
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print("len tokens:", len(tokens))
    print("sample tokens:", tokens[:30])
    # find all words with frequency less than threshold
    fdist = FreqDist(tokens)
    unknowns = dict( [ (token, count) for token, count in fdist.items()
                 if count <= THRESHOLD ] )
    # unknowns = list(filter(lambda word: fdist[word] < THRESHOLD, fdist))
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print("len unknowns", len(unknowns))
    print("sample unknowns", list(unknowns.items())[:30])
    # return lowercase, tokenized list with UNK token in place of low-freq words
    prepped = list(map(lambda word: 'UNK' if word in unknowns else word, tokens))
    """
    unks = dict([ (u, 0) for u in unknowns ])
    prepped = tokens.copy()
    for iWord, word in enumerate(tokens):
        if word in unks:
            tokens[iWord] = 'UNK'
        if iWord % 10000 == 0:
            print(".", end='')
        if iWord % 500000 == 0:
            print()
            nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
            print("====" + nowStr + "====")
    print()
    """
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print("len prepped:", len(prepped))
    print("sample prepped:", prepped[:30])
    return prepped

# read a file into a string
def file_to_str(filepath):
    with open(filepath) as f:
        str = f.read()
    return str

# read a single file and process it
def process_file(filename):
    with open(filename, 'r') as f:
        str = f.read()
    str = process_str(str)
    return str

# reads contents of files in folder into a string, then processes it
def process_folder(input_path, output_path, foldername):
    folderpath = os.path.join(input_path, foldername)
    contentList = os.listdir(folderpath)
    # contentTokens = []
    contentStr = ""
    for content in contentList:
        if content != '.DS_Store' and content != 'README':
            filepath = os.path.join(folderpath, content)
            if os.path.isfile(filepath):
                print(content)
                contentStr += file_to_str(filepath)
                # tokens = tokenize(file_to_str(filepath))
                # print(tokens)
                # contentTokens += tokens
    # return repl_infreq_w_unknown(contentTokens)
    fully_processed_str = process_str(contentStr)
    # serialize(fully_processed_str, os.path.join(output_path, foldername, foldername + '_tokens'))
    return fully_processed_str;

def serialize(tokens, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(tokens, f)

def deserialize(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
print("====" + nowStr + "====")

whitespace_test="This is is a\n test   and\nI am not     sure how it will turn      out.    "


pathToyData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/toy'
pathGutenberg = r'D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg'
pathNewsData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/news_data'
pathImdbData = r'D:/Documents/NLP/NEU_CS6120/assignment_1/imdb_data'

input_path = r'D:/Documents/NLP/NEU_CS6120/assignment_1'
input_dir = r'gutenberg'
output_path = os.path.join(input_path, "test")

# print(process_file(os.path.join(input_path, 'test.txt')))
prepped_tokens = process_folder(input_path, output_path, input_dir)
print(">>> len fully processed tokens:", len(prepped_tokens))
print("First 30 fully processed tokens:", prepped_tokens[:30])
print("Last 30 fully processed tokens:", prepped_tokens[-30:])
# print(process_folder(input_path, output_path, ""))
# print(type(file_to_str(os.path.join(input_path, 'test.txt'))))
# print(file_to_str(os.path.join(input_path, 'test.txt')))
# contentStr = "boop\n"
# contentStr += file_to_str(os.path.join(input_path, 'test.txt'))
# print(contentStr)]
# print(tokenize(whitespace_test))
# print(process_folder(os.path.join(input_path, 'gutenberg')))

nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
print("====" + nowStr + "====")
