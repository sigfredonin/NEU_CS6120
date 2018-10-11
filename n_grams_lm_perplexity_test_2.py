import n_grams_lm as lm
import nltk

def compute_perplexity(model, n, tokens, fnx_infrequent, \
        gut_pGrams, gut_pGram_UNK):
    fnx_tokens_prepped = model.infrequent_to_UNK( \
        tokens, fnx_infrequent)
    p = fnx_tokens_prepped
    fnx_n_grams = [tuple(p[i:i+n]) for i in range(len(p)-n+1)]
    fnx_n_gram_counts = nltk.FreqDist(fnx_n_grams)
    pGramsTraining = gut_pGrams
    pGramUnknown = gut_pGram_UNK
    fnx_pGrams = {}
    for gram in fnx_n_grams:
        if gram in pGramsTraining:
            pGram = pGramsTraining[gram]
        else:
            pGram = pGramUnknown
        fnx_pGrams[gram] = pGram
    fnx_perplexity = model.perplexity(list(fnx_pGrams.values()))
    print("Count %d-gram probabilities:" % (n), len(fnx_pGrams))
    print("First 30 %d-Gram probabilities" % (n))
    print(list(fnx_pGrams.items())[:30])
    print("Perplexity:", fnx_perplexity)
    return fnx_perplexity

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = lm.N_Grams_LM()
    TOO_FEW = 5

    model_gut = [ None for i in range(7) ]
    gut_pGrams = [ None for i in range(7) ]
    gut_pGram_UNK = [ None for i in range(7) ]
    for n in range(4, 5):
        print("-- train with Gutenberg data --")
        model_gut[n] = lm.N_Grams_LM()
        model_gut[n].set_n_grams_from_files(lm.pathGutenberg, n, 5, \
            SENT_SEPS=True, USE_UNK=True)
        gut_pGrams[n], gut_pGram_UNK[n] = model.alpha_smoothed_ngrams(0.1, \
            model_gut[n].tokens_UNK, n)

    for testPath in [ lm.pathToyData, lm.pathImdbData, lm.pathNewsData ]:
        for n in range(4, 5):
            print("-- test set_n_grams_from_files()- %d-grams --" % (n))
            model.set_n_grams_from_files(testPath, n, 5,
                                         SENT_SEPS=True, USE_UNK=True)
            print("Sample first 30 %d-grams found --" % (n))
            print(list(model.grams.items())[:30])
            print("Sample last 30 %d-grams found --" % (n))
            print(list(model.grams.items())[-30:])
            nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
            print()
            print("====" + nowStr + "====")

            print(" ... probabilities and perplexity --")
            fnx_infrequent = [ t for t in model.tokens
                               if t not in model_gut[n].tokens_UNK ]
            for file in model.files:
                fnx_tokens = model.tokens_in_files[file]
                compute_perplexity(model, n, fnx_tokens, fnx_infrequent, \
                    gut_pGrams[n], gut_pGram_UNK[n])

                nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
                print()
                print("====" + nowStr + "====")

