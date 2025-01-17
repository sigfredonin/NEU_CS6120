>>> 
===== RESTART: D:\Documents\NLP\NEU_CS6120\n_grams_lm_gutenberg_test.py =====
====October 09, 2018 08:42:01 AM====
-- test set_n_grams_from_files()- 1-grams --
--------------------------------------------------
-- Initialize 1-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 1-Grams |-- New --|
austen-emma.txt                     5803      5803
austen-persuasion.txt               4939      1017
austen-sense.txt                    5394       745
bible-kjv.txt                       8988      5026
blake-poems.txt                     1481       164
bryant-stories.txt                  3541       442
burgess-busterbrown.txt             1459        89
carroll-alice.txt                   2461       183
chesterton-ball.txt                 6141       815
chesterton-brown.txt                5827       365
chesterton-thursday.txt             5085       142
edgeworth-parents.txt               6754       291
melville-moby_dick.txt              9720       595
milton-paradise.txt                 6089       129
shakespeare-caesar.txt              1998       343
shakespeare-hamlet.txt              2623        82
shakespeare-macbeth.txt             2072        28
whitman-leaves.txt                  7470        85
--------------------------------------------------
Total 1-grams found: 16344
--------------------------------------------------
Sample first 30 1-grams found --
[(('[',), 131), (('Emma',), 861), (('by',), 7981), (('Jane',), 301), (('UNK',), 78681), ((']',), 131), (('I',), 29976), (('CHAPTER',), 291), (('Woodhouse',), 310), ((',',), 192339), (('handsome',), 130), (('clever',), 74), (('and',), 78725), (('rich',), 231), (('with',), 16823), (('a',), 32387), (('comfortable',), 108), (('home',), 673), (('happy',), 530), (('disposition',), 73), (('seemed',), 1083), (('to',), 46102), (('unite',), 17), (('some',), 2557), (('of',), 70031), (('the',), 125714), (('best',), 568), (('blessings',), 24), (('existence',), 44), ((';',), 27942)]
Sample last 30 1-grams found --
[(('May-be',), 7), (('disappear',), 7), (('absorb',), 14), (('recitative',), 6), (('myths',), 7), (('railroads',), 6), (('laboring',), 6), (('pennants',), 10), (('Allons',), 10), (('envelop',), 8), (('womanhood',), 6), (('Brooklyn',), 10), (('Answerer',), 6), (('growths',), 11), (('bugles',), 12), (('myriad',), 10), (('elate',), 6), (('railroad',), 6), (('unreck',), 6), (('recesses',), 6), (('Pioneers',), 28), (('en-masse',), 6), (('responding',), 7), (('dirge',), 6), (('murk',), 7), (('Vigil',), 7), (('vigil',), 6), (('aspiration',), 6), (('Unfolded',), 11), (('trumpeter',), 8)]

====October 09, 2018 08:44:55 AM====
-- test set_n_grams_from_files()- 2-grams --
--------------------------------------------------
-- Initialize 2-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 2-Grams |-- New --|
austen-emma.txt                    60174     60174
austen-persuasion.txt              38796     22476
austen-sense.txt                   50189     27258
bible-kjv.txt                     150699    129649
blake-poems.txt                     5677      2586
bryant-stories.txt                 22981     10067
burgess-busterbrown.txt             8246      3271
carroll-alice.txt                  14302      5841
chesterton-ball.txt                40977     20977
chesterton-brown.txt               37593     17270
chesterton-thursday.txt            30904     12410
edgeworth-parents.txt              63997     28263
melville-moby_dick.txt             93723     51747
milton-paradise.txt                48823     27083
shakespeare-caesar.txt             12204      6287
shakespeare-hamlet.txt             16510      7464
shakespeare-macbeth.txt            11383      4454
whitman-leaves.txt                 59969     29370
--------------------------------------------------
Total 2-grams found: 466647
--------------------------------------------------
Sample first 30 2-grams found --
[(('[', 'Emma'), 1), (('Emma', 'by'), 1), (('by', 'Jane'), 4), (('Jane', 'UNK'), 5), (('UNK', 'UNK'), 3679), (('UNK', ']'), 63), ((']', 'UNK'), 15), (('UNK', 'I'), 435), (('I', 'CHAPTER'), 1), (('CHAPTER', 'I'), 7), (('I', 'Emma'), 2), (('Emma', 'Woodhouse'), 4), (('Woodhouse', ','), 118), ((',', 'handsome'), 11), (('handsome', ','), 30), ((',', 'clever'), 3), (('clever', ','), 17), ((',', 'and'), 41329), (('and', 'rich'), 9), (('rich', ','), 35), ((',', 'with'), 2100), (('with', 'a'), 1711), (('a', 'comfortable'), 18), (('comfortable', 'home'), 3), (('home', 'and'), 22), (('and', 'happy'), 30), (('happy', 'disposition'), 1), (('disposition', ','), 20), ((',', 'seemed'), 90), (('seemed', 'to'), 403)]
Sample last 30 2-grams found --
[(('The', 'slower'), 1), (('slower', 'fainter'), 1), (('clock', 'is'), 1), ((',', 'nightfall'), 1), (('joy', "'d"), 1), ((',', 'caress'), 1), ((';', 'Delightful'), 1), (('now', 'separation'), 1), (('separation', '--'), 1), (('Long', 'indeed'), 1), ((',', 'slept'), 1), (('become', 'really'), 1), (('really', 'blended'), 1), (('blended', 'into'), 1), (('die', 'we'), 1), (('die', 'together'), 1), (('(', 'yes'), 1), (("'ll", 'remain'), 1), (('remain', 'one'), 1), (('anywhere', 'we'), 1), (('meet', 'what'), 1), (('May-be', 'we'), 1), (('learn', 'something'), 1), (('May-be', 'it'), 2), (('is', 'yourself'), 1), (('yourself', 'now'), 1), (('now', 'really'), 1), (('true', 'songs'), 1), (('Good-bye', '--'), 1), (('hail', '!'), 1)]

====October 09, 2018 08:48:05 AM====
-- test set_n_grams_from_files()- 3-grams --
--------------------------------------------------
-- Initialize 3-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 3-Grams |-- New --|
austen-emma.txt                   135258    135258
austen-persuasion.txt              76686     62297
austen-sense.txt                  106099     83743
bible-kjv.txt                     448413    428838
blake-poems.txt                     7597      6172
bryant-stories.txt                 42269     30419
burgess-busterbrown.txt            14072      9995
carroll-alice.txt                  25249     18403
chesterton-ball.txt                74850     57426
chesterton-brown.txt               67706     49159
chesterton-thursday.txt            54704     37855
edgeworth-parents.txt             143023    102107
melville-moby_dick.txt            191656    150669
milton-paradise.txt                83154     69949
shakespeare-caesar.txt             21101     16157
shakespeare-hamlet.txt             29475     21950
shakespeare-macbeth.txt            18679     13448
whitman-leaves.txt                115146     91439
--------------------------------------------------
Total 3-grams found: 1385284
--------------------------------------------------
Sample first 30 3-grams found --
[(('[', 'Emma', 'by'), 1), (('Emma', 'by', 'Jane'), 1), (('by', 'Jane', 'UNK'), 3), (('Jane', 'UNK', 'UNK'), 3), (('UNK', 'UNK', ']'), 23), (('UNK', ']', 'UNK'), 4), ((']', 'UNK', 'I'), 1), (('UNK', 'I', 'CHAPTER'), 1), (('I', 'CHAPTER', 'I'), 1), (('CHAPTER', 'I', 'Emma'), 2), (('I', 'Emma', 'Woodhouse'), 1), (('Emma', 'Woodhouse', ','), 4), (('Woodhouse', ',', 'handsome'), 1), ((',', 'handsome', ','), 6), (('handsome', ',', 'clever'), 1), ((',', 'clever', ','), 1), (('clever', ',', 'and'), 4), ((',', 'and', 'rich'), 4), (('and', 'rich', ','), 1), (('rich', ',', 'with'), 1), ((',', 'with', 'a'), 435), (('with', 'a', 'comfortable'), 1), (('a', 'comfortable', 'home'), 3), (('comfortable', 'home', 'and'), 1), (('home', 'and', 'happy'), 1), (('and', 'happy', 'disposition'), 1), (('happy', 'disposition', ','), 1), (('disposition', ',', 'seemed'), 1), ((',', 'seemed', 'to'), 22), (('seemed', 'to', 'unite'), 1)]
Sample last 30 3-grams found --
[(('off', 'and', 'UNK'), 1), (('and', 'learn', 'something'), 1), (('learn', 'something', ','), 1), (('something', ',', 'May-be'), 1), ((',', 'May-be', 'it'), 1), (('May-be', 'it', 'is'), 2), (('it', 'is', 'yourself'), 1), (('is', 'yourself', 'now'), 1), (('yourself', 'now', 'really'), 1), (('now', 'really', 'UNK'), 1), (('the', 'true', 'songs'), 1), (('true', 'songs', ','), 1), (('?', ')', 'May-be'), 1), ((')', 'May-be', 'it'), 1), (('is', 'you', 'the'), 1), (('you', 'the', 'mortal'), 1), (('the', 'mortal', 'UNK'), 1), (('mortal', 'UNK', 'really'), 1), ((',', 'turning', '--'), 1), (('turning', '--', 'so'), 1), (('--', 'so', 'now'), 1), (('so', 'now', 'finally'), 1), (('now', 'finally', ','), 1), (('finally', ',', 'Good-bye'), 1), ((',', 'Good-bye', '--'), 1), (('Good-bye', '--', 'and'), 1), (('--', 'and', 'hail'), 1), (('and', 'hail', '!'), 1), (('hail', '!', 'my'), 1), (('!', 'my', 'Fancy'), 1)]

====October 09, 2018 08:51:19 AM====
-- test set_n_grams_from_files()- 4-grams --
--------------------------------------------------
-- Initialize 4-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 4-Grams |-- New --|
austen-emma.txt                   175144    175144
austen-persuasion.txt              92898     87332
austen-sense.txt                  132036    122844
bible-kjv.txt                     689570    683094
blake-poems.txt                     8005      7714
bryant-stories.txt                 50421     45557
burgess-busterbrown.txt            16697     15240
carroll-alice.txt                  30092     27769
chesterton-ball.txt                89598     82728
chesterton-brown.txt               80123     72414
chesterton-thursday.txt            64570     57130
edgeworth-parents.txt             186339    166772
melville-moby_dick.txt            237875    221147
milton-paradise.txt                93280     89900
shakespeare-caesar.txt             24275     22671
shakespeare-hamlet.txt             34665     32086
shakespeare-macbeth.txt            21326     19513
whitman-leaves.txt                138925    131462
--------------------------------------------------
Total 4-grams found: 2060517
--------------------------------------------------
Sample first 30 4-grams found --
[(('[', 'Emma', 'by', 'Jane'), 1), (('Emma', 'by', 'Jane', 'UNK'), 1), (('by', 'Jane', 'UNK', 'UNK'), 3), (('Jane', 'UNK', 'UNK', ']'), 3), (('UNK', 'UNK', ']', 'UNK'), 3), (('UNK', ']', 'UNK', 'I'), 1), ((']', 'UNK', 'I', 'CHAPTER'), 1), (('UNK', 'I', 'CHAPTER', 'I'), 1), (('I', 'CHAPTER', 'I', 'Emma'), 1), (('CHAPTER', 'I', 'Emma', 'Woodhouse'), 1), (('I', 'Emma', 'Woodhouse', ','), 1), (('Emma', 'Woodhouse', ',', 'handsome'), 1), (('Woodhouse', ',', 'handsome', ','), 1), ((',', 'handsome', ',', 'clever'), 1), (('handsome', ',', 'clever', ','), 1), ((',', 'clever', ',', 'and'), 1), (('clever', ',', 'and', 'rich'), 1), ((',', 'and', 'rich', ','), 1), (('and', 'rich', ',', 'with'), 1), (('rich', ',', 'with', 'a'), 1), ((',', 'with', 'a', 'comfortable'), 1), (('with', 'a', 'comfortable', 'home'), 1), (('a', 'comfortable', 'home', 'and'), 1), (('comfortable', 'home', 'and', 'happy'), 1), (('home', 'and', 'happy', 'disposition'), 1), (('and', 'happy', 'disposition', ','), 1), (('happy', 'disposition', ',', 'seemed'), 1), (('disposition', ',', 'seemed', 'to'), 1), ((',', 'seemed', 'to', 'unite'), 1), (('seemed', 'to', 'unite', 'some'), 1)]
Sample last 30 4-grams found --
[(('really', 'UNK', 'me', 'to'), 1), (('me', 'to', 'the', 'true'), 1), (('to', 'the', 'true', 'songs'), 1), (('the', 'true', 'songs', ','), 1), (('true', 'songs', ',', '('), 1), (('songs', ',', '(', 'who'), 1), (('knows', '?', ')', 'May-be'), 1), (('?', ')', 'May-be', 'it'), 1), ((')', 'May-be', 'it', 'is'), 1), (('May-be', 'it', 'is', 'you'), 1), (('it', 'is', 'you', 'the'), 1), (('is', 'you', 'the', 'mortal'), 1), (('you', 'the', 'mortal', 'UNK'), 1), (('the', 'mortal', 'UNK', 'really'), 1), (('mortal', 'UNK', 'really', 'UNK'), 1), (('UNK', 'really', 'UNK', ','), 1), (('really', 'UNK', ',', 'turning'), 1), (('UNK', ',', 'turning', '--'), 1), ((',', 'turning', '--', 'so'), 1), (('turning', '--', 'so', 'now'), 1), (('--', 'so', 'now', 'finally'), 1), (('so', 'now', 'finally', ','), 1), (('now', 'finally', ',', 'Good-bye'), 1), (('finally', ',', 'Good-bye', '--'), 1), ((',', 'Good-bye', '--', 'and'), 1), (('Good-bye', '--', 'and', 'hail'), 1), (('--', 'and', 'hail', '!'), 1), (('and', 'hail', '!', 'my'), 1), (('hail', '!', 'my', 'Fancy'), 1), (('!', 'my', 'Fancy', '.'), 1)]

====October 09, 2018 08:54:38 AM====
-- test set_n_grams_from_files()- 5-grams --
--------------------------------------------------
-- Initialize 5-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 5-Grams |-- New --|
austen-emma.txt                   187241    187241
austen-persuasion.txt              96842     95392
austen-sense.txt                  138912    136367
bible-kjv.txt                     813661    812677
blake-poems.txt                     8112      8070
bryant-stories.txt                 53078     51518
burgess-busterbrown.txt            17757     17406
carroll-alice.txt                  31863     31400
chesterton-ball.txt                94417     92729
chesterton-brown.txt               83733     81592
chesterton-thursday.txt            67691     65374
edgeworth-parents.txt             201493    195563
melville-moby_dick.txt            250709    246584
milton-paradise.txt                95143     94552
shakespeare-caesar.txt             24946     24668
shakespeare-hamlet.txt             35922     35422
shakespeare-macbeth.txt            21947     21550
whitman-leaves.txt                145783    144302
--------------------------------------------------
Total 5-grams found: 2342407
--------------------------------------------------
Sample first 30 5-grams found --
[(('[', 'Emma', 'by', 'Jane', 'UNK'), 1), (('Emma', 'by', 'Jane', 'UNK', 'UNK'), 1), (('by', 'Jane', 'UNK', 'UNK', ']'), 3), (('Jane', 'UNK', 'UNK', ']', 'UNK'), 1), (('UNK', 'UNK', ']', 'UNK', 'I'), 1), (('UNK', ']', 'UNK', 'I', 'CHAPTER'), 1), ((']', 'UNK', 'I', 'CHAPTER', 'I'), 1), (('UNK', 'I', 'CHAPTER', 'I', 'Emma'), 1), (('I', 'CHAPTER', 'I', 'Emma', 'Woodhouse'), 1), (('CHAPTER', 'I', 'Emma', 'Woodhouse', ','), 1), (('I', 'Emma', 'Woodhouse', ',', 'handsome'), 1), (('Emma', 'Woodhouse', ',', 'handsome', ','), 1), (('Woodhouse', ',', 'handsome', ',', 'clever'), 1), ((',', 'handsome', ',', 'clever', ','), 1), (('handsome', ',', 'clever', ',', 'and'), 1), ((',', 'clever', ',', 'and', 'rich'), 1), (('clever', ',', 'and', 'rich', ','), 1), ((',', 'and', 'rich', ',', 'with'), 1), (('and', 'rich', ',', 'with', 'a'), 1), (('rich', ',', 'with', 'a', 'comfortable'), 1), ((',', 'with', 'a', 'comfortable', 'home'), 1), (('with', 'a', 'comfortable', 'home', 'and'), 1), (('a', 'comfortable', 'home', 'and', 'happy'), 1), (('comfortable', 'home', 'and', 'happy', 'disposition'), 1), (('home', 'and', 'happy', 'disposition', ','), 1), (('and', 'happy', 'disposition', ',', 'seemed'), 1), (('happy', 'disposition', ',', 'seemed', 'to'), 1), (('disposition', ',', 'seemed', 'to', 'unite'), 1), ((',', 'seemed', 'to', 'unite', 'some'), 1), (('seemed', 'to', 'unite', 'some', 'of'), 1)]
Sample last 30 5-grams found --
[(('UNK', 'me', 'to', 'the', 'true'), 1), (('me', 'to', 'the', 'true', 'songs'), 1), (('to', 'the', 'true', 'songs', ','), 1), (('the', 'true', 'songs', ',', '('), 1), (('true', 'songs', ',', '(', 'who'), 1), (('songs', ',', '(', 'who', 'knows'), 1), (('who', 'knows', '?', ')', 'May-be'), 1), (('knows', '?', ')', 'May-be', 'it'), 1), (('?', ')', 'May-be', 'it', 'is'), 1), ((')', 'May-be', 'it', 'is', 'you'), 1), (('May-be', 'it', 'is', 'you', 'the'), 1), (('it', 'is', 'you', 'the', 'mortal'), 1), (('is', 'you', 'the', 'mortal', 'UNK'), 1), (('you', 'the', 'mortal', 'UNK', 'really'), 1), (('the', 'mortal', 'UNK', 'really', 'UNK'), 1), (('mortal', 'UNK', 'really', 'UNK', ','), 1), (('UNK', 'really', 'UNK', ',', 'turning'), 1), (('really', 'UNK', ',', 'turning', '--'), 1), (('UNK', ',', 'turning', '--', 'so'), 1), ((',', 'turning', '--', 'so', 'now'), 1), (('turning', '--', 'so', 'now', 'finally'), 1), (('--', 'so', 'now', 'finally', ','), 1), (('so', 'now', 'finally', ',', 'Good-bye'), 1), (('now', 'finally', ',', 'Good-bye', '--'), 1), (('finally', ',', 'Good-bye', '--', 'and'), 1), ((',', 'Good-bye', '--', 'and', 'hail'), 1), (('Good-bye', '--', 'and', 'hail', '!'), 1), (('--', 'and', 'hail', '!', 'my'), 1), (('and', 'hail', '!', 'my', 'Fancy'), 1), (('hail', '!', 'my', 'Fancy', '.'), 1)]

====October 09, 2018 08:57:55 AM====
-- test set_n_grams_from_files()- 6-grams --
--------------------------------------------------
-- Initialize 6-gram model for the files in directory
--   D:/Documents/NLP/NEU_CS6120/assignment_1/gutenberg
-- with tokens that occur 5 or fewer times
-- replaced by 'UNK'
--------------------------------------------------
----+- Files -+----+----+----| 6-Grams |-- New --|
austen-emma.txt                   190364    190364
austen-persuasion.txt              97639     97255
austen-sense.txt                  140667    139964
bible-kjv.txt                     869690    869580
blake-poems.txt                     8156      8147
bryant-stories.txt                 54041     53329
burgess-busterbrown.txt            18203     18137
carroll-alice.txt                  32650     32585
chesterton-ball.txt                96056     95666
chesterton-brown.txt               84780     84217
chesterton-thursday.txt            68703     67992
edgeworth-parents.txt             206437    204822
melville-moby_dick.txt            253681    252734
milton-paradise.txt                95472     95297
shakespeare-caesar.txt             25120     25073
shakespeare-hamlet.txt             36214     36124
shakespeare-macbeth.txt            22097     22004
whitman-leaves.txt                147787    147487
--------------------------------------------------
Total 6-grams found: 2440777
--------------------------------------------------
Sample first 30 6-grams found --
[(('[', 'Emma', 'by', 'Jane', 'UNK', 'UNK'), 1), (('Emma', 'by', 'Jane', 'UNK', 'UNK', ']'), 1), (('by', 'Jane', 'UNK', 'UNK', ']', 'UNK'), 1), (('Jane', 'UNK', 'UNK', ']', 'UNK', 'I'), 1), (('UNK', 'UNK', ']', 'UNK', 'I', 'CHAPTER'), 1), (('UNK', ']', 'UNK', 'I', 'CHAPTER', 'I'), 1), ((']', 'UNK', 'I', 'CHAPTER', 'I', 'Emma'), 1), (('UNK', 'I', 'CHAPTER', 'I', 'Emma', 'Woodhouse'), 1), (('I', 'CHAPTER', 'I', 'Emma', 'Woodhouse', ','), 1), (('CHAPTER', 'I', 'Emma', 'Woodhouse', ',', 'handsome'), 1), (('I', 'Emma', 'Woodhouse', ',', 'handsome', ','), 1), (('Emma', 'Woodhouse', ',', 'handsome', ',', 'clever'), 1), (('Woodhouse', ',', 'handsome', ',', 'clever', ','), 1), ((',', 'handsome', ',', 'clever', ',', 'and'), 1), (('handsome', ',', 'clever', ',', 'and', 'rich'), 1), ((',', 'clever', ',', 'and', 'rich', ','), 1), (('clever', ',', 'and', 'rich', ',', 'with'), 1), ((',', 'and', 'rich', ',', 'with', 'a'), 1), (('and', 'rich', ',', 'with', 'a', 'comfortable'), 1), (('rich', ',', 'with', 'a', 'comfortable', 'home'), 1), ((',', 'with', 'a', 'comfortable', 'home', 'and'), 1), (('with', 'a', 'comfortable', 'home', 'and', 'happy'), 1), (('a', 'comfortable', 'home', 'and', 'happy', 'disposition'), 1), (('comfortable', 'home', 'and', 'happy', 'disposition', ','), 1), (('home', 'and', 'happy', 'disposition', ',', 'seemed'), 1), (('and', 'happy', 'disposition', ',', 'seemed', 'to'), 1), (('happy', 'disposition', ',', 'seemed', 'to', 'unite'), 1), (('disposition', ',', 'seemed', 'to', 'unite', 'some'), 1), ((',', 'seemed', 'to', 'unite', 'some', 'of'), 1), (('seemed', 'to', 'unite', 'some', 'of', 'the'), 1)]
Sample last 30 6-grams found --
[(('UNK', 'me', 'to', 'the', 'true', 'songs'), 1), (('me', 'to', 'the', 'true', 'songs', ','), 1), (('to', 'the', 'true', 'songs', ',', '('), 1), (('the', 'true', 'songs', ',', '(', 'who'), 1), (('true', 'songs', ',', '(', 'who', 'knows'), 1), (('songs', ',', '(', 'who', 'knows', '?'), 1), (('(', 'who', 'knows', '?', ')', 'May-be'), 1), (('who', 'knows', '?', ')', 'May-be', 'it'), 1), (('knows', '?', ')', 'May-be', 'it', 'is'), 1), (('?', ')', 'May-be', 'it', 'is', 'you'), 1), ((')', 'May-be', 'it', 'is', 'you', 'the'), 1), (('May-be', 'it', 'is', 'you', 'the', 'mortal'), 1), (('it', 'is', 'you', 'the', 'mortal', 'UNK'), 1), (('is', 'you', 'the', 'mortal', 'UNK', 'really'), 1), (('you', 'the', 'mortal', 'UNK', 'really', 'UNK'), 1), (('the', 'mortal', 'UNK', 'really', 'UNK', ','), 1), (('mortal', 'UNK', 'really', 'UNK', ',', 'turning'), 1), (('UNK', 'really', 'UNK', ',', 'turning', '--'), 1), (('really', 'UNK', ',', 'turning', '--', 'so'), 1), (('UNK', ',', 'turning', '--', 'so', 'now'), 1), ((',', 'turning', '--', 'so', 'now', 'finally'), 1), (('turning', '--', 'so', 'now', 'finally', ','), 1), (('--', 'so', 'now', 'finally', ',', 'Good-bye'), 1), (('so', 'now', 'finally', ',', 'Good-bye', '--'), 1), (('now', 'finally', ',', 'Good-bye', '--', 'and'), 1), (('finally', ',', 'Good-bye', '--', 'and', 'hail'), 1), ((',', 'Good-bye', '--', 'and', 'hail', '!'), 1), (('Good-bye', '--', 'and', 'hail', '!', 'my'), 1), (('--', 'and', 'hail', '!', 'my', 'Fancy'), 1), (('and', 'hail', '!', 'my', 'Fancy', '.'), 1)]

====October 09, 2018 09:01:15 AM====
>>> 