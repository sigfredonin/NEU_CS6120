"""
Run Viterbi decoder on Assignment 1 Problem 4.5 data.
Train HMM with brown data subset

Test data file:
< sentence ID =1>
The
expense
and
time
involved
are
astronomical
.
<EOS>
< sentence ID =2>
However
,
we
sent
a
third
vessel
out
,
a
much
smaller
and
faster
one
than
the
first
two
.
<EOS>
...

Output File Format:

< sentence ID=1>
word, tag
word, tag
...
word, tag
<EOS>
< sentence ID=1>
word, tag
word, tag
...
word, tag
<EOS>

"""
import pos_hmm_viterbi as pos
import pos_hmm_bigram as hmm
import os
from collections import defaultdict
from datetime import datetime

# ------------------------------------------------------------------------
# NLP 2ed Fig 5.18 Example ---
# ------------------------------------------------------------------------


pT_5_18 = {
  '$S'   : [ ('VB', .019 ), ('TO', .0043 ), ('NN', .041  ), ('PPSS', .067  )],
  'VB'   : [ ('VB', .0038), ('TO', .035  ), ('NN', .047  ), ('PPSS', .0070 )],
  'TO'   : [ ('VB', .83  ), ('TO', .0    ), ('NN', .00047), ('PPSS', .0    )],
  'NN'   : [ ('VB', .004 ), ('TO', .016  ), ('NN', .087  ), ('PPSS', .0045 )],
  'PPSS' : [ ('VB', .23  ), ('TO', .00079), ('NN', .0012 ), ('PPSS', .00014)],
}

pE_5_18 = {
  '$S'   : [ ('I', .0 ), ('want', .0     ), ('to', .0 ), ('race', .0     )],
  'VB'   : [ ('I', .0 ), ('want', .0093  ), ('to', .0 ), ('race', .00012 )],
  'TO'   : [ ('I', .0 ), ('want', .0     ), ('to', .99), ('race', .0     )],
  'NN'   : [ ('I', .0 ), ('want', .000054), ('to', .0 ), ('race', .00057 )],
  'PPSS' : [ ('I', .37), ('want', .0     ), ('to', .0 ), ('race', .0     )],
}
pE_5_18['$S'] += [ ('<s>', 1.0) ]

o_5_18 = [ 'I', 'want', 'to', 'race' ]

# ------------------------------------------------------------------------
# Test File I/O ---
# ------------------------------------------------------------------------

def get_sentences_from_file(input_path, filename):
    filepath = os.path.join(input_path, filename)
    sentences = []
    sentence = []
    sentence_ids = []
    ss = '< sentence ID ='
    eos = '<EOS>'
    with open(filepath) as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            if line[:len(ss)] == ss:
                sentence_ids.append(line)
            elif line[:len(eos)] == eos:
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line)
    return sentences, sentence_ids

def output_tagged_sentences(output_path, filename, tagged_sentences, sentence_ids):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outFN = filename + "_tagged_" + timestamp + ".txt"
    with open(os.path.join(output_path, outFN), 'w') as f:
        for i, sentence in enumerate(tagged_sentences):
            f.write(sentence_ids[i] + '\n')
            for word, tag in sentence:
                f.write(word + ", " + tag + '\n')
            f.write('<EOS>\n')


# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

def test_viterbi(model, pT, pTU, pE, pEU):
    print("Begin test with Fig 5.18 example")
    vd = pos.POS_HMM_Viterbi(pT, pTU, pE, pEU, DEBUG=False, VERBOSE=True)
    print("Tags - tag set ---")
    print(vd.tags)
    print("dT - transmission probabilities ---")
    print("count:", len(vd.dT), "\ndT:", vd.dT)
    print("dE - emission probabilities ---")
    print("count:", len(vd.dE), "\ndT:", vd.dE)
    words, tags, probabilities = vd.decode(o_5_18)
    print(' Results '.center(49, '-'))
    print("Obvservations")
    print(words)
    print("Most probable tags")
    print(tags)
    print("Sequence probabilities")
    print(probabilities)
    swt = zip(words, tags)
    sent, tagged_sent = model._assemble_sentence(swt)
    print()
    print("Sentence ----")
    print(sent)
    print("Tagged sentence ----")
    print(tagged_sent)
    print("Probability of sentence:", probabilities[-1])

def tag_sentences_in_file(testPath, testFileName, outFileName, pT, pTU, pE, pEU):
    header = " Analyze %s/%s sentences " % (testPath, testFileName)
    print(header.center(49, '-'))
    print("Output:", testPath + '/' + outFileName)
    vd = pos.POS_HMM_Viterbi(pT, pTU, pE, pEU, DEBUG=False, VERBOSE=False)
    sentences, sentence_ids = get_sentences_from_file(testPath, testFileName)
    tagged_sentences = []
    results = []
    for iS, sentence_tokens in enumerate(sentences):
        print(sentence_ids[iS].center(40, '-'))
        words, tags, probabilities = vd.decode(sentence_tokens)
        swt = list(zip(words, tags))
        sent, tagged_sent = model._assemble_sentence(swt)
        print(20*"=")
        print(sent)
        print(tagged_sent)
        tagged_sentences += [ swt ]
        results += [ list(zip(words, tags, probabilities, sent, tagged_sent)) ]
    output_tagged_sentences(testPath, outFileName, tagged_sentences, sentence_ids)

if __name__ == '__main__':

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = hmm.POS_HMM_BiGram(DEBUG=False)

    pU = defaultdict(lambda: 1e-128)
    test_viterbi(model, pT_5_18, pU, pE_5_18, pU)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Initialize HMM with data from ToyPOS brown extract.")
    model.init(hmm.pathToyPOS, TOO_FEW=1)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    test_viterbi(model, model.pTagTrans, model.pTransUnseen, \
                        model.pTagEmiss, model.pTransUnseen)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = "D:/Documents/NLP/NEU_CS6120/assignment_1"
    testFN = "science_sample.txt"
    outFN = "science_sample_toyPOS"
    pT = model.pTagTrans
    pTU = model.pTransUnseen
    pE = model.pTagEmiss
    pEU = model.pEmissUnseen
    tag_sentences_in_file(testPath, testFN, outFN, pT, pTU, pE, pEU)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Initialize HMM with data from brown corpus.")
    model.init(hmm.pathBrownData, TOO_FEW=3)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    test_viterbi(model, model.pTagTrans, model.pTransUnseen, \
                        model.pTagEmiss, model.pTransUnseen)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = "D:/Documents/NLP/NEU_CS6120/assignment_1"
    testFN = "science_sample.txt"
    outFN = "science_sample_brown_data"
    pT = model.pTagTrans
    pTU = model.pTransUnseen
    pE = model.pTagEmiss
    pEU = model.pEmissUnseen
    tag_sentences_in_file(testPath, testFN, outFN, pT, pTU, pE, pEU)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
