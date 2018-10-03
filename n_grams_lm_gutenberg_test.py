import n_grams_lm as lm

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    testPath = lm.pathGutenberg
    model_n = lm.N_Grams_LM()
    SENT_SEPS = True        # Use <s> </s> delimiters
    if testPath == lm.pathGutenberg:
        SENT_SEPS = False   # Don't use <s> </s> delimiters
    for n in range(1, 7):
        print("-- test set_n_grams_from_files()- %d-grams --" % (n))
        model_n.set_n_grams_from_files(testPath, n, 5,
                                       SENT_SEPS=False, USE_UNK=True)
        print("Sample first 30 %d-grams found --" % (n))
        print(list(model_n.grams.items())[:30])
        print("Sample last 30 %d-grams found --" % (n))
        print(list(model_n.grams.items())[-30:])
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print()
        print("====" + nowStr + "====")
