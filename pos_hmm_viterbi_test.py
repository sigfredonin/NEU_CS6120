"""
Test Viterbi decoder.
"""
import pos_hmm_bigram as pos

# ------------------------------------------------------------------------
# NLP 2ed Fig 5.18 Example ---
# ------------------------------------------------------------------------

pT_5_18 = {
  '<s>'  : [ ('VB', .019 ), ('TO', .0043 ), ('NN', .041  ), ('PPSS', .067  )],
  'VB'   : [ ('VB', .0038), ('TO', .035  ), ('NN', .047  ), ('PPSS', .0070 )],
  'TO'   : [ ('VB', .83  ), ('TO', .0    ), ('NN', .00047), ('PPSS', .0    )],
  'NN'   : [ ('VB', .004 ), ('TO', .016  ), ('NN', .087  ), ('PPSS', .0045 )],
  'PPSS' : [ ('VB', .23  ), ('TO', .00079), ('NN', .0012 ), ('PPSS', .00014)],
}

pE_5_18 = {
  '<s>'  : [ ('I', .0 ), ('want', .0     ), ('to', .0 ), ('race', .0     )],
  'VB'   : [ ('I', .0 ), ('want', .0093  ), ('to', .0 ), ('race', .00012 )],
  'TO'   : [ ('I', .0 ), ('want', .0     ), ('to', .99), ('race', .0     )],
  'NN'   : [ ('I', .0 ), ('want', .000054), ('to', .0 ), ('race', .00057 )],
  'PPSS' : [ ('I', .37), ('want', .0     ), ('to', .0 ), ('race', .0     )],
}

o_5_18 = [ 'I', 'want', 'to', 'race']

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Begin test with Fig 5.18 example")
    pTagTrans = pT_5_18
    pTagEmiss = pE_5_18
    vd = pos.POS_HMM_Viterbi(pTagTrans, pTagEmiss)

    print("Tags - tag set ---")
    print(vd.tags)
    print("dT - transmission probabilities ---")
    print(vd.dT)
    print("dE - emission probabilities ---")
    print(vd.dE)
