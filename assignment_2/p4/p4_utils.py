"""
p4_utils.py
Utilities to support the solution of Assigment 2 Problem 4, Summary Evaluation

In this question, you will design classifiers to evaluate summary quality based
on its non-redundancy and fluency. You will be given a training data and a test
data, both in the following format:
1. Column 1: summary for a news article
2. Column 2: non-redundancy score
  [a score that indicates the conciseness of the summary]
  Non-redundancy scores range from -1 [highly redundant] to 1 [no redundancy].
3. Column 3: fluency
  [a score that indicates whether a sentence is grammatically correct in the summary]
  Fluency can range from -1 [grammatically poor] to 1 [fluent and grammatically correct].

The file is in .csv spreadsheet format - the summary is a quoted string, and it
is followed by the scores, with commas between, as in ...

"The 41-year-old was the first female official on the ▃ level and the first to
 retain a bowl game. She was the first female official on the ▃ level and the
 first to retain a bowl game, the 2009 little caesars pizza bowl in Detroit.",-1,0

Sig Nin
October 26, 2018
"""

import os
import re
import numpy as np

# ------------------------------------------------------------------------
# Input summary data ---
# ------------------------------------------------------------------------

PATH_TRAIN = "data/a2_p4_train_set.csv"
PATH_TEST = "data/a2_p4_test_set.csv"

def get_text_from_file(filePath):
    """
    Read all of the text from a file.
    This is expected to be a file containing the summaries,
    one quoted sentence per line, with non-redundancy and fluency
    scores at the end, separated by commas from the sentence and
    each other.
    Example:
      "The movie really stunk ! The movie proved bad bad .",-1,0
    """
    with open(filePath) as f:
        text = f.read()
    return text

re_summary = re.compile(r'^\"(.+)\",(.+),(.+)$', re.MULTILINE)

def get_summary_training_data(text):
    records = re_summary.findall(text)
    summaries, non_redundancies, fluencies = zip(*records)
    np_non_redundancies_float = np.array(non_redundancies).astype(np.float)
    np_fluencies_float = np.array(fluencies).astype(np.float)
    return summaries, np_non_redundancies_float, np_fluencies_float
