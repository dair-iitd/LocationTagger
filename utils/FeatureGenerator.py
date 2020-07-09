import re
import os
import sys
import json
import tqdm
import spacy
import pickle
import argparse
import functools
import itertools
from rake_nltk import Rake
from collections import defaultdict
# from gensim.models import Word2Vec
from nltk.parse import CoreNLPParser
# from nltk.parse.corenlp import CoreNLPDependencyParser

from utils import common

class FeatureBuilder:
    def __init__(self, options):
        self.options = options
        self.descriptive_phrases = set(open(self.options.descriptive_phrases_path, "r").readlines())
        self.word_vectors = {word: "G" + groupid for word, groupid in [line.strip().split(" ") for line in open(options.word_vectors_path, "r").readlines()]}
        self.stop_words = [line.strip().lower() for line in open(options.stop_words_path, "r").readlines() if not line.startswith("//")]

    def buildFeatures(self, text):
        try:
            tokens = list(self.options.tok_parser.tokenize(text))
            pos_tags = list(self.options.pos_tagger.tag(tokens))
            ner_tags = list(self.options.ner_tagger.tag(tokens))
            tokens = list([token for token, tag in pos_tags])
        except:
            error = ""
            error += "Please start the StanfordCoreNLPServer!" + "\n"
            error += "cd /home/goelshashank007/Documents/btp/java/stanford-corenlp-full-2018-10-05/" + "\n"
            error += "Start the CoreNLP Server: java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,pos,ner,depparse -status_port 9000 -port 9000 -timeout 15000" + "\n"
            error += "Stop the CoreNLP Server: wget \"localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`\" -O -"
            raise Exception(error)

        self.features = [[token] for token in tokens]

        def setFirstCharCaps():
            for i, token in enumerate(tokens):
                if(token[0].isupper()):
                    if((i == 0) or (tokens[i - 1] == "?") or (tokens[i - 1] == ".") or (tokens[i - 1] == "!")):
                        continue
                    self.features[i].append("FIRST_CHAR_CAPS")

        def setNumbers():
            for i, token in enumerate(tokens):
                if(re.match(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", token)):
                    self.features[i].append("IS_NUMBER")

        def setPrevPosTag():
            self.features[0].append("NA")
            for i, (token, tag) in enumerate(pos_tags[:-1]):
                self.features[i + 1].append(tag.lower())

        def setNoun():
            for i, (token, tag) in enumerate(pos_tags):
                tag = tag.lower()
                if(tag.startswith("nn")):
                    self.features[i].append("NOUN_P" if tag == "nnp" else "NOUN")

        def setAdjectives():
            for i, (token, tag) in enumerate(pos_tags):
                tag = tag.lower()
                if(tag.startswith("jj")):
                    self.features[i].append("ADJ")

        def setNerTags():
            for i, (token, tag) in enumerate(ner_tags):
                if(tag != "O"):
                    self.features[i].append(tag)

        def setDescriptivePhrases():
            i = 0
            while(i < len(tokens) - 1):
                token = tokens[i]
                tag = pos_tags[i][1].lower()

                k = 1
                while(1):
                    token_plus_k = tokens[i + k]
                    tag_plus_k = pos_tags[i + k][1].lower()
                    if(len(token_plus_k) > 1):
                        break
                    k += 1
                    if(i + k == len(tokens)):
                        break

                if(i + k < len(tokens)):
                    if(((tag == "jj") and (tag_plus_k in ["nn", "nns", "nnp"])) or ((token + " " + token_plus_k).lower() in self.descriptive_phrases)):
                        self.features[i].append("DESC_PHRASE")
                        self.features[i + k].append("DESC_PHRASE")
                        i += k

                i += 1

        def setWhInducedTarget():
            for i in range(len(tokens)):
                if(i < len(tokens) - 3):
                    if(pos_tags[i][1].lower().startswith("w") and (not tokens[i].lower() == "who") and pos_tags[i + 1][1].lower() == "to" and pos_tags[i + 2][1].lower().startswith("v") or (pos_tags[i][1].lower().startswith("n") and pos_tags[i + 1][1].lower() == "to" and pos_tags[i + 2][1].lower().startswith("v"))):
                        self.features[i].append("WH_INDUCED_TARGET")
                        self.features[i + 1].append("WH_INDUCED_TARGET")
                        self.features[i + 2].append("WH_INDUCED_TARGET")

                        if (i < len(tokens) - 4):
                            if(pos_tags[i + 3][1].lower().startswith("n")):
                                self.features[i + 3].append("WH_INDUCED_TARGET")

        def setTypeIndicatorBasedOnVerb():
            try:
                text = " ".join(tokens).lower()
                self.options.rake.extract_keywords_from_text(text)
                phrases = [phrase for score, phrase in self.options.rake.get_ranked_phrases_with_scores() if score > 4.0]
                positions = [(text[:re.search(phrase, text).start()].count(" ") + index) for phrase in phrases for index in range(phrase.count(" ") + 1) if phrase in text]
                for i in positions:
                    self.features[i].append("TYPE_INDICATOR_BASED_ON_VB")
            except:
                pass

        def setWordAfterNumber():
            for i in range(1, len(tokens)):
                if("IS_NUMBER" in self.features[i - 1]):
                    self.features[i].append("wordAfterNUM")

        def setTypeIndPlusWhCombo():
            for i in range(len(tokens)):
                if("TYPE_INDICATOR_BASED_ON_VB" in self.features[i] and "WH_INDUCED_TARGET" in self.features[i]):
                    self.features[i].append("TYPE_INDICATOR_VB_PLUS_WH")

        def setWordVector():
            for i, token in enumerate(tokens):
                if(token in self.word_vectors):
                    self.features[i].append(self.word_vectors[token])

        def setWordCount():
            word_count = defaultdict(int)
            for i, token in enumerate(tokens):
                if(token in self.stop_words):
                    continue
                word_count[token] += 1

            for i, token in enumerate(tokens):
                if(token in word_count):
                    self.features[i].append("NUM_%d" % word_count[token])

        setFirstCharCaps()
        setNumbers()
        setPrevPosTag()
        setNoun()
        setAdjectives()
        setNerTags()
        setDescriptivePhrases()
        setWhInducedTarget()
        setTypeIndicatorBasedOnVerb()
        setWordAfterNumber()
        setTypeIndPlusWhCombo()
        setWordVector()
        setWordCount()

        return self.features

class FeatureProcessor:
    def __init__(self, options):
        self.options = options

    def processFeatures(self, features):
        self.features = features

        def sentenceSplitter():
            self.sentences = [list(x[1]) for x in itertools.groupby(self.features, lambda l: l[0] in set(['.', '?','!',';']))]
            self.sentences = list(map(lambda x: x[0] + x[1], itertools.zip_longest(self.sentences[::2], self.sentences[1::2], fillvalue = [])))

        def processSentences1():
            allowed = set(['ADJ', 'NOUN', 'PROPN'])
            notallowed = set(['PRP$', 'PDT', 'WDT', 'WP$'])

            for i in range(len(self.sentences)):
                sentence = self.sentences[i]
                type_indicator_words = set([features[0] for features in sentence if any("TYPE_INDICATOR" in feature for feature in features)])
                if(not type_indicator_words):
                	continue

                text = " ".join([features[0] for features in sentence])
                document = self.options.nlp(text)
                for index, (token, features) in enumerate(zip(document, sentence)):
                	if((token.pos_ in allowed) and (not token.text in type_indicator_words) and (not token.tag_ in notallowed)):
                		head = token
                		while(head.head != head):
                			if(head.text in type_indicator_words):
                				sentence[index].append("PREDICTED_ATTR_BY_FEAT")
                				break
                			head = head.head

                self.sentences[i] = sentence

        # def processSentences2():
        #     def checkSimilar(word1, word2):
        #         try:
        #             if(word1 == word2):
        #                 return 1
        #             for word, score in self.gensim_model.wv.similar_by_vector(word1):
        #                 if(word2 == word):
        #                 	return 1
        #             return 0
        #         except:
        #         	return 0
        #
        #     for i, sentence in enumerate(self.sentences):
        #         string = " ".join([features[0] for features in sentence])
        #         search_for_to = 0
        #         in_location_indicators, going_location_indicators = {}, {}
        #         in_word_in_consideration, going_word_in_consideration = "", ""
        #
        #         dependency_parse = self.options.dep_parser.raw_parse(string).__next__().triples()
        #         for head, tag, body in dependency_parse:
        #             if(checkSimilar(head[0], "travelling") and body[1] == "TO"):
        #                 search_for_to = 1
        #             if(search_for_to and (head[1] == "TO") and (tag in ["pobj", "dep"])):
        #             	search_for_to = 0
        #             	if("NN" in body[1]):
        #             		going_location_indicators[body[0]] = body[1]
        #             		going_word_in_consideration = body[0]
        #             elif(going_word_in_consideration and (head[0] == going_word_in_consideration)):
        #             	if(tag in ["nn", "dobj", "dep"]):
        #             		going_location_indicators[body[0]] = body[1]
        #             if((checkSimilar(head[0], "in") or checkSimilar(head[0], "near")) and (tag in ["pobj", "dep"])):
        #             	if("NN" in body[1]):
        #             		in_location_indicators[body[0]] = body[1]
        #             		in_word_in_consideration = body[0]
        #             elif(in_word_in_consideration and (head[0] == in_word_in_consideration)):
        #             	if(tag in ["nn", "dobj", "dep"]):
        #             		in_location_indicators[body[0]] = body[1]
        #
        #         for j, features in enumerate(sentence):
        #             if((features[0] in in_location_indicators) and (" DATE " not in features)):
        #             	features.append("IN_LOC_INDICATOR")
        #             	features.append(in_location_indicators[features[0]])
        #             if((features[0] in going_location_indicators) and (" DATE " not in features)):
        #             	features.append("GOING_LOC_INDICATOR")
        #             	features.append(going_location_indicators[features[0]])
        #             sentence[j] = features
        #
        #         self.sentences[i] = sentence

        sentenceSplitter()
        processSentences1()
        # processSentences2()

        self.features = list(itertools.chain.from_iterable(self.sentences))

        return self.features

def getFeatures(sentences, labels = None):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["descriptive_phrases_path"] = project_root_path / "data/features/DescriptivePhrases.txt"
    defaults["word_vectors_path"] = project_root_path / "data/features/WordVectors.txt"
    defaults["stop_words_path"] = project_root_path / "data/features/StopWords.txt"
    # defaults["gensim_model_path"] = project_root_path / "data/features/Gensim.model"
    defaults["allowed_features_file_path"] = project_root_path / "data/features/features.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptive_phrases_path", type = str, default = str(defaults["descriptive_phrases_path"]))
    parser.add_argument("--word_vectors_path", type = str, default = str(defaults["word_vectors_path"]))
    parser.add_argument("--stop_words_path", type = str, default = str(defaults["stop_words_path"]))
    # parser.add_argument("--gensim_model_path", type = str, default = str(defaults["gensim_model_path"]))
    parser.add_argument("--allowed_features_file_path", type = str, default = str(defaults["allowed_features_file_path"]))
    options = parser.parse_args("")

    options.rake = Rake()
    options.nlp = spacy.load("en_core_web_sm")
    # options.gensim = Word2Vec.load(options.gensim_model_path)
    options.tok_parser = CoreNLPParser(url = "http://localhost:9000")
    options.pos_tagger = CoreNLPParser(url = "http://localhost:9000", tagtype = "pos")
    options.ner_tagger = CoreNLPParser(url = "http://localhost:9000", tagtype = "ner")
    # options.dep_parser = CoreNLPDependencyParser(url = "http://localhost:9000")

    allowed_features = set([line.strip() for line in open(options.allowed_features_file_path, "r").readlines()])

    featureBuilder = FeatureBuilder(options)
    featureProcessor = FeatureProcessor(options)

    features = []
    bar = tqdm.tqdm(total = len(sentences))

    if(labels is None):
        labels = ["O"] * len(sentences)

    for sentence, ilabels in zip(sentences, labels):
        ifeatures = featureBuilder.buildFeatures(sentence)
        ifeatures = featureProcessor.processFeatures(ifeatures)

        item = []
        for token_features, label in zip(ifeatures, ilabels):
            item.append(token_features[0] + " " + " ".join([feature for feature in token_features[1:] if feature in allowed_features]) + " " + label)

        features.append("\n".join(item))
        bar.update()

    return "\n\n".join(features)
