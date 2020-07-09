Start the CoreNLP Server: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000
Stop the CoreNLP Server: wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -

Run the feature builder
$ cd <path>/stanford-corenlp-full-2018-10-05/
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload depparse -status_port 9000 -port 9000 -timeout 15000
$ cd scratch/BiLSTM-CCM/
$ python generate/v2/generateFeatureFile.py
$ wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -

# If the /tmp/corenlp.shutdown then
$ lsof -i:9000
# Find <pid> of the command java.
$ kill <pid>
