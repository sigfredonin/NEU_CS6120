@ ECHO Parse %1
java -mx500m -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat "wordsAndTags,penn,typedDependencies" -maxLength 50 englishPCFG.ser.gz %1