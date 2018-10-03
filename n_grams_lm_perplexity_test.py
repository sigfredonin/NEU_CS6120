import n_grams_lm as lm

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = lm.N_Grams_LM()
    TOO_FEW = 5

    for testPath in [ lm.pathToyData, lm.pathImdbData, lm.pathNewsData ]:

        USE_SENT_SEPS = True        # Use <s> </s> delimiters
        if testPath == lm.pathGutenberg:
            USE_SENT_SEPS = False   # Don't use <s> </s> delimiters

        for n in range(1, 7):

            print("-- test set_n_grams_from_files()- %d-grams --" % (n))
            model.set_n_grams_from_files(testPath, n, 5,
                                         SENT_SEPS=USE_SENT_SEPS, USE_UNK=True)
            print("Sample first 30 %d-grams found --" % (n))
            print(list(model.grams.items())[:30])
            print("Sample last 30 %d-grams found --" % (n))
            print(list(model.grams.items())[-30:])

            nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
            print()
            print("====" + nowStr + "====")

            print(" ... probabilities and perplexity --")
            fnx_pGrams = model.alpha_smoothed_ngrams(0.1, \
                model.tokens_UNK, model.grams)
            fnx_perplexity = model.perplexity(list(fnx_pGrams.values()))
            print("Count %d-gram probabilities:" % (n), len(fnx_pGrams))
            print("First 30 %d-Gram probabilities" % (n))
            print(list(fnx_pGrams.items())[:30])
            print("Perplexity:", fnx_perplexity)

            nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
            print()
            print("====" + nowStr + "====")

