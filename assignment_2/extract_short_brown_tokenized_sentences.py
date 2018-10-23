"""
extract_short_brown_tokenized_sentences

Extract sentences with no more than 50 words
from the tokenized Brown files.

Sig Nin
October 17, 2018
"""

import re
import os
from datetime import datetime
import split_tokenized_sentences as selector

# -----------------------------------------------------------------------------
# Select short sentences from the files in a directory
# -----------------------------------------------------------------------------

def select_and_write_short_sentences(dirPath, outDir, MAX_WORDS):
	sents_in_files = selector.get_short_sentences_in_files(dirPath, MAX_WORDS)
	for file, sents_in_file in sents_in_files.items():
		sents, short_sents = sents_in_file
		print("Found %d sentences no longer than %d words out of %d sentences in %s." % (len(short_sents), MAX_WORDS, len(sents), file))
		for i, sent_words in enumerate(short_sents):
			sent, words = sent_words
			print("%3d: (%3d) '%s'" % (i, len(words), sent))
			print((3+3+3+2)*' ', words)
		if len(short_sents) > 0:
			short_sents_in_file, sent_words_in_file = zip(*short_sents)
		else:
			short_sents_in_file = [ '' ]
		selector.output_short_sentences(outDir, file, short_sents_in_file)

# -----------------------------------------------------------------------------
# Select short sentences from the files in the Brown_tokenized_text directory
# -----------------------------------------------------------------------------

if __name__ == '__main__':

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
	print("====" + nowStr + "====")

	print("Test short sentence selection from files in a directory".center(80, '-'))

	MAX_WORDS = 50
	
	dirPath = "Brown_tokenized_text"
	print("Selecting short sentences in the files in %s" % dirPath)

	timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
	outDir = "./short/" + dirPath + timestamp
	os.makedirs(outDir, exist_ok=True)
	print("Output will be written to '%s'" % outDir)

	sents_in_files = selector.extract_short_sentences_in_files(dirPath, outDir, MAX_WORDS)
	for file, counts in sents_in_files.items():
		count_all, count_short = counts
		print("%-20s: %d of %d" % (file, count_short, count_all))

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
	print("====" + nowStr + "====")

