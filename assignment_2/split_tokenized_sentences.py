"""
split_tokenized_sentences

Split a tokenized file into sentences at " . ", " ! ", or " ? ".
Find the sentences no longer than a given number of words.

Sig Nin
October 17, 2018
"""

import re
import os
from datetime import datetime

DEBUG=False

# REGEX for splitting a text into sentences.
# Note: This expresion handles multiple sentence enders,
#       e.g. ? ? or ?? or ?? !!, but it does not make
#       an ending quote after the period a part of the sentence,
#       as in: She said, "I will not do it."
# re_for_sents = r'((?:(?:[^\s\:!\.\?]+[^\s\:!\?]*\s+)*)+[\.!\?](?:\s*[\.!\?])*)\s+'
re_for_sents = r'((?:(?:[^ \:!\?\.]+[^ \:!\?]* +)*)+[\.!\?](?: *[\'\.!\?])*) +'
re_sents = re.compile(re_for_sents)

# REX for splitting a tokenized sentence into tokens.
re_for_tokens = r'(\S+)'
re_tokens = re.compile(re_for_tokens)

# Punctuation marks, for excluding punctuation
# when counting tokens that are words.
punctuation = { '``', "''", '"', "'", ',', ';', ':', '.', '!', '?' }

# Sentence ending tokens
sentence_enders = { '.', '!', '?', '!!', '??', '--' }

# -----------------------------------------------------------------------------
# Select short sentences from the files in a directory
# -----------------------------------------------------------------------------

def get_short_sentences_RE(text, MAX_WORDS=50):
	"""
	Extract sentences with no more than MAX_WORDS words
	from a tokenized text that is in a single string.
	The tokens are words and puctuation separated by whitespace.
	The punctuation characters are "``", "''", '"', "'", ',', ';', ':', '.', '!', and '?'.
	Sentences are assumed to end with '.', '!', or '?' tokens.
	Words are tokens that are not punctuation.
	"""
	sents = re_sents.findall(text)
	short_sents = []
	for sent in sents:
		print("--> '%s'" % (sent))
		tokens = re_tokens.findall(sent)
		words = [ token for token in tokens if token not in punctuation ]
		if len(words) <= MAX_WORDS:
			short_sents.append( ( sent, words, ) )
	return sents, short_sents


def get_short_sentences_from_tokens(tokens, sents, MAX_WORDS=50):
	"""
	Extract sentences with no more than MAX_WORDS words
	from the tokens and sentences.
	"""
	short_sents = []
	for iSent, sent in enumerate(sents):
		tokens_in_sent = tokens[iSent]
		words = [ token for token in tokens_in_sent if token not in punctuation ]
		if len(words) <= MAX_WORDS:
			short_sents.append( ( sent, words, ) )
	return sents, short_sents

def get_short_sentences(text, MAX_WORDS=50):
	"""
	Extract sentences with no more than MAX_WORDS words
	from a tokenized text that is in a single string.
	The tokens are words and puctuation separated by whitespace.
	The punctuation characters are "``", "''", '"', "'", ',', ';', ':', '.', '!', and '?'.
	Sentences are assumed to end with '.', '!', or '?' tokens.
	Words are tokens that are not punctuation.
	"""
	tokens = text.split(' ')
	sents = []
	tokens_in_sents = []
	sent = ''
	tokens_in_sent = []
	prev_token = ''
	for iToken, token in enumerate(tokens):
		if prev_token in sentence_enders \
		   and token not in sentence_enders and token != "''":
			print("[%3d]--> '%s'" % (len(tokens_in_sent), sent))
			sents.append(sent)
			tokens_in_sents.append(tokens_in_sent)
			sent = ''
			tokens_in_sent = []
		sent += token + " "
		tokens_in_sent.append(token)
		prev_token = token
	print("[LAST][%d]--> '%s' %s" % (len(tokens_in_sent), sent, tokens_in_sent))
	if sent != ' ':
		sents.append(sent)
		tokens_in_sents.append(tokens_in_sent)
	return get_short_sentences_from_tokens(tokens_in_sents, sents, MAX_WORDS)

def get_short_sentences_in_file(filePath, MAX_WORDS=50):
	with open(filePath) as f:
		text = f.read()
	return get_short_sentences(text, MAX_WORDS)
	
def get_short_sentences_in_files(dirPath, MAX_WORDS=50):
	files = sorted(os.listdir(dirPath))
	sents_in_files = {}
	for file in files:
		print("=====>> %s" % file)
		filePath = os.path.join(dirPath, file)
		sents_words = get_short_sentences_in_file(filePath, MAX_WORDS)
		sents_in_files[file] = sents_words
	return sents_in_files

def output_sentences(output_path, filename, sentences, ADD_NL=False):
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	outFN = filename + timestamp + ".txt"
	with open(os.path.join(output_path, outFN), 'w') as f:
		f.write(' ')
		for sentence in sentences:
			f.write(sentence)
			f.write(' ')
			if ADD_NL:
				f.write('\n')
	
def extract_short_sentences_in_files(dirPath, outDir, MAX_WORDS=50):
	files = sorted(os.listdir(dirPath))
	count_sents_in_files = {}
	for file in files:
		print("=====>> %s" % file)
		filePath = os.path.join(dirPath, file)
		sents, short_sents = get_short_sentences_in_file(filePath, MAX_WORDS)
		if DEBUG:
			for i, sent_words in enumerate(short_sents):
				sent, words = sent_words
				print("%3d: (%3d) '%s'" % (i, len(words), sent))
				print((3+3+3+2)*' ', words)
		if len(short_sents) > 0:
			short_sents_in_file, sent_words_in_file = zip(*short_sents)
		else:
			short_sents_in_file = [ '' ]
		output_sentences(outDir, file + "_short_", short_sents_in_file)
		output_sentences(outDir, file + "_short_Lines", short_sents_in_file, ADD_NL=True)
		output_sentences(outDir, file + "_all_lines", sents, ADD_NL=True)
		count_sents_in_files[file] = ( len(sents), len(short_sents), )
	return count_sents_in_files

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if __name__ == '__main__':

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")
	
	DEBUG=True
	
	print("Sentence REGEX: r'%s'" % re_for_sents)
	print("Tokens REGEX: r'%s'" % re_for_tokens)
	print("Punctuation marks:", punctuation)
	
	text = "It was a windy , dark and stormy night . Mr. D. Kaplan woke frightened . "
	sents = re_sents.findall(text)
	print("sent = '%s'" % text)
	print("re_sents.findall(text) = '%s'" % sents)
	print(["%d: '%s'" % (i, sent) for i, sent in enumerate(sents) ])
	
	timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
	outDir = "./short/test" + timestamp
	os.makedirs(outDir, exist_ok=True)
	print("Output will be written to '%s'" % outDir)
	
	print("Test short sentence selection".center(80, '-'))

	text = \
		"""The Office of Business Economics ( OBE ) of the U.S. Department of Commerce provides basic measures of the national economy and current analysis of short-run changes in the economic situation and business outlook . It develops and analyzes the national income , balance of international payments , and many other business indicators . Such measures are essential to its job of presenting business and Government with the facts required to meet the objective of expanding business and improving the operation of the economy . Contact For further information contact Director , Office of Business Economics , U.S. Department of Commerce , Washington 25 , D.C. . Printed material Economic information is made available to businessmen and economists promptly through the monthly Survey Of Current Business and its weekly supplement . This periodical , including weekly statistical supplements , is available for $4 per year from Commerce Field Offices or Superintendent of Documents , U.S. Government Printing Office , Washington 25 , D.C. . Technical assistance to small business community The Small Business Administration ( SBA ) provides guidance and advice on sources of technical information relating to small business management and research and development of products . Small business management Practical management problems and their suggested solutions are dealt with in a series of SBA publications . These publications , written especially for the managers or owners of small businesses , indirectly aid in community development programs . They are written by specialists in numerous types of business enterprises , cover a wide range of subjects , and are directed to the needs and interests of the small firm . Is it really so ? Oh , my gosh ! Oh ! Oh ! Really ? """
	MAX_WORDS = 20
	sents, short_sents = get_short_sentences(text, MAX_WORDS)
	print("Found %d sentences no longer than %d words out of %d sentences." % (len(short_sents), MAX_WORDS, len(sents)))
	for i, sent_words in enumerate(short_sents):
		sent, words = sent_words
		print("%3d: (%3d) '%s'" % (i, len(words), sent))
		print((3+3+3+2)*' ', words)
	output_sentences(outDir, "government_short_", sents)
		
	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

	print("Test word counting, excluding punctuation".center(80, '-'))
	
	text_with_punctuation = \
		"""Oh ! Oh ! Really ? ! ? Really ?? !! ?? Really : I couldn't say , why do you want to know , and why does anyone else ? She said , `` I don't know . '' I'm sure that I don't know , either , nor does anyone else that I know of . Mr. D. D. Kaplan , M.D. , of New Paltz , N.Y. , insists it is the greek letter ' kappa ' , of all things . """

	MAX_WORDS = 100
	sents, short_sents = get_short_sentences(text_with_punctuation, MAX_WORDS)
	print("Found %d sentences no longer than %d words out of %d sentences." % (len(short_sents), MAX_WORDS, len(sents)))
	for i, sent_words in enumerate(short_sents):
		sent, words = sent_words
		print("%3d: (%3d) '%s'" % (i, len(words), sent))
		print((3+3+3+2)*' ', words)
	output_sentences(outDir, "test_punctuation_short_", sents)

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

	print("Test short sentence selection from a file".center(80, '-'))

	MAX_WORDS = 3
	filePath = "toy_tokenized/samiam.txt"
	print("Selecting short sentences in %s" % filePath)
	sents, short_sents = get_short_sentences_in_file(filePath, MAX_WORDS)
	print("Found %d sentences no longer than %d words out of %d sentences." % (len(short_sents), MAX_WORDS, len(sents)))
	for i, sent_words in enumerate(short_sents):
		sent, words = sent_words
		print("%3d: (%3d) '%s'" % (i, len(words), sent))
		print((3+3+3+2)*' ', words)
	output_sentences(outDir, "samiam_short_", sents)

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

	print("Test short sentence selection from files in a directory".center(80, '-'))

	MAX_WORDS = 25
	dirPath = "toy_tokenized"
	print("Selecting short sentences in the files in %s" % dirPath)
	sents_in_files = extract_short_sentences_in_files(dirPath, outDir, MAX_WORDS)

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

