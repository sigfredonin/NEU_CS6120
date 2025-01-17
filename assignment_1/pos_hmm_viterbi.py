"""
NEU CS6120 Assignment 1
Problem 4 POS Tagging - Hidden Markov Model

Viterbi Decoder

Implements the Viterbi decoding algorithm in NLP ed3 Fig. 8.5
to derive the most probable tag sequence for a sentence.

Note: Rather than initialize the decoder by processing the state
transition from <s> to the first word, initialized the decoder
to the <start> state.  This simplifies the code in that the first
state update is the same as all the rest.

The state (tag) probabilities are set to:
    '$S' : 1.0
    rest : 0.0

The back pointers are all set to None.

Sig Nin
07 Oct 2018
"""

import numpy as np
import os
import re
from collections import defaultdict

_DEBUG_ = False
_VERBOSE_ = False

TOK_SS = '<s>'  # start sentence
TAG_SS = '$S'
TOK_ES = '</s>' # end sentence
TAG_ES = 'S$'

# ------------------------------------------------------------------------
# Helper Class - Viterbi Decoder ---
# ------------------------------------------------------------------------

class POS_HMM_Viterbi:
    """
    HMM Viterbi POS Decoder

    Initialize with transmission and emission probabilities,
    indexed by tag:
        pTagTrans: { prev_tag : [ ( tag, probability), ...], ...}
        pTagEmiss: { curr_tag : [ ( word, probability), ...], ...}
    """
# ------------------------------------------------------------------------
# Class constructor ---
# ------------------------------------------------------------------------

    def _probability_LUT(self, pTagProbs, pProbsUnseen):
        """
        Create a transition or emission probability lookup table.
        Input:
            pTagProbs: { t_i-1 : [ ( t_i, P(t_i-1, t_i), ...], ...}
                   or: ( t_i   : [ ( w_i, P(w_i | t_i),  ...], ...}
            pProbsUnseen: { t_i-1 : probability_of_unseen, ... }
                      or: ( t_i   : probability_of_unseen, ... )
        Output:
            dT : { prev_tag : { tag : probability, ...}, ...}
        """
        probUnseen = pProbsUnseen[None]
        print("_probability_LUT.probUnseen:", probUnseen)
        dP = defaultdict(lambda: defaultdict(lambda: probUnseen))
        for prior, pList in pTagProbs.items():
            probUnseenGivenPrior = pProbsUnseen[prior]
            dV = defaultdict(lambda: probUnseenGivenPrior)
            for value, probability in pList:
                dV[value] = probability
            dP[prior] = dV
        return dP

    def _init(self, pTagTrans, pTransUnseen, pTagEmiss, pEmissUnseen):
        # tag set
        self.tags = set(sorted(pTagTrans))
        if self.tags != set(sorted(pTagEmiss)):
            msg = ""
            print("ERROR: transmission and emission probabilities",
                  "are not for the same tag set.")
            return
        # state transition probabilities P(t_i-1, t_i)
        self.dT = self._probability_LUT(pTagTrans, pTransUnseen)
        # word emission probabilities P( w_i | t_i )
        self.dE = self._probability_LUT(pTagEmiss, pEmissUnseen)

    def set_DEBUG(self, DEBUG):
        _DEBUG_ = DEBUG

    def set_VERBOSE(self, VERBOSE):
        _VERBOSE_ = VERBOSE

    def __init__(self, pTagTrans, pTransUnseen, pTagEmiss, pEmissUnseen, \
                 DEBUG=False, VERBOSE=False):
        self.set_DEBUG(DEBUG)
        self.set_VERBOSE(VERBOSE)
        self._init(pTagTrans, pTransUnseen, pTagEmiss, pEmissUnseen)

# ------------------------------------------------------------------------
# Viterbi decoding ---
# ------------------------------------------------------------------------

    def _find_max_results(self, pS, bS):
        probabilities = [ p for tag, p in pS.items() ]
        pMax = max(probabilities)
        tagsMax = [ tag for tag, p in pS.items() if p == pMax ]
        if len(tagsMax) > 1:
            print("Warning: there are %d tags with max p = %f" \
                % (len(tagsMax), pMax) )
        tMax = tagsMax[0]
        return pMax, tMax

    def _find_max(self, tag, pE, pS_prev):
        pMax = 0.0
        tMax = None
        for prev_tag in self.tags: # iterate over previous tags
            pPrev = pS_prev[prev_tag]   # Viterbi[i-1, prev_tag]
            pTs = self.dT[prev_tag]     # { tag : P(prev_tag, tag)}
            pT = pTs[tag]               # P(prev_tag, tag)
            p = pPrev * pT * pE         # ? Viterbi(i, tag)
            if p > pMax:
                pMax = p
                tMax = prev_tag
        return pMax, tMax

    def _step(self, word, pS):
        pS_prev = pS            # Viterbi[i-1, tag], previous time step
        pS = {}                 # Viterbi[i, tag], this time step
        bS = {}                 # backpointer[i, tag], this time step
        for tag in self.tags:   # iterate over all possible POS tags
            # probability of this word given this tag
            pEs = self.dE[tag]      # word probabilities for this tag
            pE = pEs[word]          # P(word | tag)
            # find previous tag that gives highest probability for tag
            pMax, tMax = self._find_max(tag, pE, pS_prev)
            pS[tag] = pMax
            bS[tag] = tMax
        print("--- %s ---" % word)
        if _VERBOSE_:
            print("pS:", pS)
            print("bS:", bS)
        return pS, bS

    def _backtrace(self, observations, viterbi, backpointer):
        """
        # find the maximum probability and corresponding tag in each time step
        # and follow the back pointers to determine the most probable tags
        """
        print("Backtrace".center(47, '-'))
        most_probable_tags = []
        max_probabilities = []
        for iV, pS in reversed(list(enumerate(viterbi))):
            bS = backpointer[iV]
            word = observations[iV]
            if _VERBOSE_:
                print("----------")
                print("iV:", iV)
                print("word:", word)
                print("pS:", pS)
                print("bS:", bS)
            pMax, tMax = self._find_max_results(pS, bS)
            if _VERBOSE_:
                print("tMax, pMax:", tMax, pMax)
            most_probable_tags = [ tMax ] + most_probable_tags
            max_probabilities = [ pMax ] + max_probabilities
        return observations, most_probable_tags, max_probabilities

    def decode(self, observations):
        """
        Find the most probable POS tag assignment for
        a given list of observed tokens.
        Inputs:
            observations: [ token, ... ]
        """
        # Initialize with state at start of sentence
        pS = { t : 1.0 if t == TAG_SS else 0.0 for t in self.tags }
        bS = { t : None for t in self.tags }
        viterbi = []      # Viterbi[iW, tag], at start
        backpointer = []  # backpointer[iW, tag], at start
        print("--- %s ---" % TOK_SS)
        if _VERBOSE_:
            print("pS:", pS)
            print("bS:", bS)
        # iterate over observations, starting with first word
        for word in observations:
            pS, bS = self._step(word, pS)
            viterbi += [ pS ]      # Viterbi[iW, tag], this time step
            backpointer += [ bS ]  # backpointer[iW, tag], this time step
        # termination: transition to end of sentence
        observations, most_probable_tags, max_probabilities = \
            self._backtrace(observations, viterbi, backpointer)
        print(" Tagging ".center(49, '-'))
        print("words:", observations)
        print(" tags:", most_probable_tags)
        print("probs:", max_probabilities)
        return observations, most_probable_tags, max_probabilities
