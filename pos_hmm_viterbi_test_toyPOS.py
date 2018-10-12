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

# ------------------------------------------------------------------------
# NLP 2ed Fig 5.18 Example ---
# ------------------------------------------------------------------------

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
    with open(os.path.join(output_path, filename + "_tagged.txt"), 'w') as f:
        for i, sentence in enumerate(tagged_sentences):
            f.write(sentence_ids[i] + '\n')
            for word, tag in sentence:
                f.write(word + ", " + tag + '\n')
            f.write('<EOS>\n')


# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Initialize HMM with data from ToyPOS brown extract.")
    model = hmm.POS_HMM_BiGram()
    model.init(hmm.pathToyPOS, TOO_FEW=1)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Begin test with Fig 5.18 example")
    vd = pos.POS_HMM_Viterbi(model.pTagTrans,model.pTagEmiss)

    print("Tags - tag set ---")
    print(vd.tags)
    print("dT - transmission probabilities ---")
    print("count:", len(vd.dT), "\ndT:", list(vd.dT.items())[:30], "\n...")
    print("dE - emission probabilities ---")
    print("count:", len(vd.dE), "\ndT:", list(vd.dE.items())[:30], "\n...")

    words, tags, probabilities = vd.decode(o_5_18)

    print('Results'.center(47, '-'))
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

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = "D:/Documents/NLP/NEU_CS6120/assignment_1"
    testFileName = "science_sample.txt"
    outFileName = "science_sample_tagged.txt"
    sentences, sentence_ids = get_sentences_from_file(testPath, testFileName)
    tagged_sentences = []
    results = []
    for iS, sentence_tokens in enumerate(sentences):
        words, tags, probabilities = vd.decode(sentence_tokens)
        sent, tagged_sent = model._assemble_sentence(swt)
        print(sentence_ids[iS].center(40, '-'))
        print(sent)
        print(tagged_sent)
        tagged_sentences += [ zip(words, tags) ]
        results += [ list(zip(words, tags, probabilities, sent, tagged_sent)) ]
    output_tagged_sentences(testPath, filename, tagged_sentences, sentence_ids)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
