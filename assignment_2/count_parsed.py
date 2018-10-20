"""
CS 6120 Assignment 2 Problem 1
Count verbs, sentences

Verbs: count instances of
	'(VB ', '(VBD ', '(VBG ', '(VBN ', '(VBZ ' ... in parse tree
	'/VB ', '/VBD ', '/VBG ', '/VBN ', '/VBZ ' ... in tagged sentences
Sentences: count instances of
	'(ROOT' ... in parse tree

Sig Nin
October 18, 2018
"""

import re
import os
from datetime import datetime
from nltk import FreqDist

# -----------------------------------------------------------------------------
# Count Verbs
# -----------------------------------------------------------------------------

verb_tags = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'ALL', )

# ... in the parse trees
re_pt_VB_ = re.compile(r'\(VB.')	# VB* all forms
re_pt_VBb = re.compile(r'\(VB ')	# VB  base form
re_pt_VBD = re.compile(r'\(VBD ')	# VBD past tense
re_pt_VBG = re.compile(r'\(VBG ')	# VBG gerund or past participle
re_pt_VBN = re.compile(r'\(VBN ')	# VBN past participle
re_pt_VBP = re.compile(r'\(VBP ')	# VBP present tense, except 3rd Per Sing
re_pt_VBZ = re.compile(r'\(VBZ ')	# VBZ present tense, 3rd Person Singular

# ... in the tagged sentences
re_ts_VB_ = re.compile(r'/VB.')		# VB* all forms
re_ts_VBb = re.compile(r'/VB ')		# VB  base form
re_ts_VBD = re.compile(r'/VBD ')	# VBD past tense
re_ts_VBG = re.compile(r'/VBG ')	# VBG gerund or past participle
re_ts_VBN = re.compile(r'/VBN ')	# VBN past participle
re_ts_VBP = re.compile(r'/VBP ')	# VBP present tense, except 3rd Per Sing
re_ts_VBZ = re.compile(r'/VBZ ')	# VBZ present tense, 3rd Person Singular

def count_verbs(text):
	"""
	Count verbs, in the parse tree and in the tagged sentences.
	The counts must agree.
	"""

	# ... in the parse trees
	pt_VB_  = len(re_pt_VB_.findall(text))	# VB* all forms
	pt_VBb  = len(re_pt_VBb.findall(text))	# VB  base form
	pt_VBD  = len(re_pt_VBD.findall(text))	# VBD past tense
	pt_VBG  = len(re_pt_VBG.findall(text))	# VBG gerund or past participle
	pt_VBN  = len(re_pt_VBN.findall(text))	# VBN past participle
	pt_VBP  = len(re_pt_VBP.findall(text))	# VBP present tense, except 3rd Per Sing
	pt_VBZ  = len(re_pt_VBZ.findall(text))	# VBZ present tense, 3rd Person Singular

	# ... in the tagged sentences
	ts_VB_  = len(re_ts_VB_.findall(text))	# VB* all forms
	ts_VBb  = len(re_ts_VBb.findall(text))	# VB  base form
	ts_VBD  = len(re_ts_VBD.findall(text))	# VBD past tense
	ts_VBG  = len(re_ts_VBG.findall(text))	# VBG gerund or past participle
	ts_VBN  = len(re_ts_VBN.findall(text))	# VBN past participle
	ts_VBP  = len(re_ts_VBP.findall(text))	# VBP present tense, except 3rd Per Sing
	ts_VBZ  = len(re_ts_VBZ.findall(text))	# VBZ present tense, 3rd Person Singular

	# ... collect all counts
	pt_verb_counts = ( pt_VBb, pt_VBD, pt_VBG, pt_VBN, pt_VBP, pt_VBZ, pt_VB_, )
	ts_verb_counts = ( ts_VBb, ts_VBD, ts_VBG, ts_VBN, ts_VBP, ts_VBZ, ts_VB_, )

	# check agreement
	if (pt_verb_counts != ts_verb_counts):
		print("ERROR: counts from parse tree do not agree with counts from tagged sentences.")

	# return all counts
	return ( pt_verb_counts, ts_verb_counts, )

# -----------------------------------------------------------------------------
# Count Parsed Sentences
# -----------------------------------------------------------------------------

re_pt_ROOT = re.compile(r'^\(ROOT', re.MULTILINE)
re_dp_ROOT = re.compile(r'^root\(ROOT', re.MULTILINE)

def count_parsed_sentences(text):
	"""
	Count sentences processed by the parser, as indicated by:
	(1) the count of "(ROOT" in the parse trees
	(2) the oount of "root(ROOT" in the dependency tagging
	"""
	pt_ROOT = len(re_pt_ROOT.findall(text))
	dp_ROOT = len(re_dp_ROOT.findall(text))
	return pt_ROOT, dp_ROOT

# -----------------------------------------------------------------------------
# Count Prepositions
# -----------------------------------------------------------------------------

re_pt_IN = re.compile(r'\(IN ([^\(\)]*)\)', re.MULTILINE)
re_pt_TO = re.compile(r'\(TO ([^\(\)]*)\)', re.MULTILINE)

def count_prepositions(text):
	"""
	Count the prepositions in the parse trees and dependency tags.
	In the parse tree, there are two preposition tags:
		IN	preposition, except 'to'
		TO	preposition, 'to'
	In the dependency tagging, prepositions are tagged according to
	their function relative to other words and phrases, so it is
	difficult to identify them.
	"""
	# parse tree, tag 'IN'
	pt_IN = re_pt_IN.findall(text)
	pt_IN_lc = [ s.lower() for s in pt_IN ]
	
	# parse tree, tag 'TO'
	pt_TO = re_pt_TO.findall(text)
	pt_TO_lc = [ s.lower() for s in pt_TO ]
	
	# compile counts and most frequent prepositions
	count_total = len(pt_IN_lc) + len(pt_TO_lc)
	fd_TO_IN = FreqDist(pt_TO_lc + pt_IN_lc)
	count_unique = len(fd_TO_IN)
	most_common_3 = fd_TO_IN.most_common(3)
	
	# return results and frequency distribution
	return count_total, count_unique, most_common_3, fd_TO_IN

# -----------------------------------------------------------------------------
# Read results file, Write counts
# -----------------------------------------------------------------------------

def get_results(filePath):
	with open(filePath) as f:
		text = f.read()
	return text

def write_counts(outPath, filePath, verb_counts, root_counts, prep_data):
	pt_verb_counts, ts_verb_counts = verb_counts
	pt_root_count, ts_root_count = root_counts
	prep_count_total, prep_count_unique, prep_top_3, prep_fd = prep_data

	with open(outPath, 'a') as f:
		f.write("\nFile: %s\n" % filePath)
		f.write("%-10s: %8s %8s\n" % ('verb', 'trees', 'sents'))
		f.write("%-10s--%8s-%8s\n" % (10*'-', 8*'-', 8*'-'))

		for i, tag in enumerate(verb_tags):
			pt_count = pt_verb_counts[i]
			ts_count = ts_verb_counts[i]
			if i == len(verb_tags) - 1:
				f.write("%-10s--%8s-%8s\n" % (10*'-', 8*'-', 8*'-'))
			f.write("%-10s: %8d %8d\n" % (tag, pt_count, ts_count))

		f.write("Parsed sentences -------- \n")
		f.write("        Parse trees : %8d\n" % pt_root_count)
		f.write("    Depencency tags : %8d\n" % ts_root_count)
		f.write("Prepositions -------------\n")
		f.write("    Total IN and TO : %8d\n" % prep_count_total)
		f.write("   Unique IN and TO : %8d\n" % prep_count_unique)
		f.write(" Top 3 prepositions : %s\n"  % prep_top_3)
			
def write_totals(outPath, total_roots, all_fd):
	print("<--- %s %8d %s" % (outPath, total_roots, all_fd.most_common(3)))
	with open(outPath, 'a') as f:
		f.write("\nTotal sentences parsed over all files: %8d\n" % total_roots)
		f.write("Top 3 prepositions over all files: %s\n" % all_fd.most_common(3))


def write_counts_files(dirPath, outDir):
	files = sorted(os.listdir(dirPath))
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	outPath = os.path.join(outDir, "counts_" + timestamp + ".txt")
	print("--> %s" % outPath)

	total_roots = 0
	all_fd = FreqDist([])
	for filename in files:
		filePath = os.path.join(dirPath, filename)
		print()
		print("<-- %s" % filePath)
		text = get_results(filePath)
		verb_counts = count_verbs(text)
		root_counts = count_parsed_sentences(text)
		prep_data = count_prepositions(text)
		write_counts(outPath, filePath, verb_counts, root_counts, prep_data)
		pt_verb_counts, ts_verb_counts = verb_counts
		

		print("Tagged Verbs --------------")
		print("       Parse trees :",pt_verb_counts)
		print("  Tagged sentences :",ts_verb_counts)

		pt_root_count, ts_root_count = root_counts
		print("Parsed Sentences ----------")
		print("       Parse trees :",pt_root_count)
		print("   Dependency tags :",ts_root_count)

		prep_count_total, prep_count_unique, prep_top_3, prep_fd = prep_data
		print("Prepositions --------------")
		print("    Tota; IN and TO : %8d" % prep_count_total)
		print("   Unique IN and TO : %8d" % prep_count_unique)
		print(" Top 3 prepositions : %s"  % prep_top_3)
		
		total_roots += pt_root_count
		all_fd += prep_fd
		write_totals(outPath, total_roots, all_fd)

	print("Total sentences parsed over all files: %8d" % total_roots)
	print("Top 3 prepositions over all files: %s" % all_fd.most_common(3))

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if __name__ == '__main__':

	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

	dirPath = "output"
	files = sorted(os.listdir(dirPath))
	
	filename = files[0]
	filePath = os.path.join(dirPath, filename)
	text = get_results(filePath)
	verb_counts = count_verbs(text)
	pt_verb_counts, ts_verb_counts = verb_counts
	root_counts = count_parsed_sentences(text)
	pt_root_count, ts_root_count = root_counts
	prep_data = count_prepositions(text)
	prep_count_total, prep_count_unique, prep_top_3, prep_fd = prep_data
	
	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")
	
	print("File: %s" % filePath)
	print("%-10s: %8s %8s" % ('verb', 'trees', 'sents'))
	print("%-10s--%8s-%8s" % (10*'-', 8*'-', 8*'-'))
	for i, tag in enumerate(verb_tags):
		pt_verb_count = pt_verb_counts[i]
		ts_verb_count = ts_verb_counts[i]
		if i == len(verb_tags) - 1:
			print("%-10s--%8s-%8s" % (10*'-', 8*'-', 8*'-'))
		print("%-10s: %8d %8d" % (tag, pt_verb_count, ts_verb_count))
	print("Parsed Sentences  -------")
	print("        Parse trees : %8d" % pt_root_count)
	print("    Dependency tags : %8d" % ts_root_count)
	print("Prepositions -------------")
	print("    Tota; IN and TO : %8d" % prep_count_total)
	print("   Unique IN and TO : %8d" % prep_count_unique)
	print(" Top 3 prepositions : %s"  % prep_top_3)
	print()
	
	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	outDir = "results"
	outPath = os.path.join(outDir, filename + "_counts_" + timestamp + ".txt")
	write_counts(outPath, filePath, verb_counts, root_counts, prep_data)

	write_counts_files(dirPath, outDir)
	
	nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S.%f %p")
	print("====" + nowStr + "====")
