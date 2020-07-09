import os
import sys
import copy
import json
import nltk
import tqdm
import string
import argparse
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from . import common

class LocationProcessor:
    def __init__(self):
        project_root_path = common.getProjectRootPath()

        self.punctuations = string.punctuation
        self.stopwords = set(stopwords.words("english"))
        self.acronyms = list(map(lambda x: x.strip(), open(project_root_path / "data/utils/sanitizer/acronyms.txt").readlines()))
        self.cities = list(map(lambda x: x.strip(), open(project_root_path / "data/utils/sanitizer/cities.txt").readlines()))
        self.states = list(map(lambda x: x.strip(), open(project_root_path / "data/utils/sanitizer/states.txt").readlines()))
        self.countries = list(map(lambda x: x.strip(), open(project_root_path / "data/utils/sanitizer/countries.txt").readlines()))
        self.continents = list(map(lambda x: x.strip(), open(project_root_path / "data/utils/sanitizer/continents.txt").readlines()))

    @staticmethod
    def checkInstance(instance):
        text = " ".join([token.split("_")[0] for token in instance.split(" ") if (token not in self.punctuations and token not in self.stopwords)]).lower()
        if(len(text) < 4):
            return False
        if(any(list(map(lambda x: text in x, [self.acronyms, self.cities, self.states, self.countries, self.continents])))):
            return False
        return True

    @staticmethod
    def process(instances):
        processed_instances = []
        for instance in instances:
            if(LocationProcessor.checkInstance(copy.deepcopy(instance))):
                processed_instances.append(instance)

        return processed_instances

def getSpans(tokens, pos_tags, iob_tags):
    conlltags = [("%s_%d" % (token, index), pos_tag, iob_tag) for index, (token, pos_tag, iob_tag) in enumerate(zip(tokens, pos_tags, iob_tags))]
    tree = nltk.chunk.conlltags2tree(conlltags)

    instances = set()
    for subtree in tree:
        if(type(subtree) == nltk.Tree):
            instance = " ".join([token_index for token_index, pos_tag in subtree.leaves()])
            instances.add(instance)

    return list(instances)

def generateSanitizedPredictions(options):
    data = json.load(open(options.input_file_path))

    sentences = sorted(list({item["question"] for item in data}))
    features = open(options.features_file_path).read().split("\n\n")
    predictions = open(options.predictions_file_path).read().split("\n\n")

    tokens = [[line.split(" ", 1)[0] for line in ifeatures.split("\n")] for ifeatures in features]
    predictions = [ipredictions.split("\n") for ipredictions in predictions]

    processor = LocationProcessor()
    pos_tagger = CoreNLPParser(url = "http://localhost:9000", tagtype = "pos")

    d = {}
    print("Generating Sanitized Preditions/Tagged Locations")
    bar = tqdm.tqdm(total = len(tokens))
    for sentence, itokens, ipredictions in zip(sentences, tokens, predictions):
        ipos_tags = list(pos_tagger.tag(itokens))
        itagged_locations = getSpans(itokens, ipos_tags, ipredictions)
        if(options.process):
            itagged_locations = processor.process(itagged_locations)
        d[sentence] = {"question_pos": " ".join(["%s_%d" % (token, i) for i, token in enumerate(itokens)]), "tagged_locations": itagged_locations}
        bar.update()
    bar.close()

    for item in data:
        item["question_pos"] = d[item["question"]]["question_pos"]
        item["tagged_locations"] = d[item["question"]]["tagged_locations"]

    common.dumpJSON(data, options.output_file_path)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["input_file_path"] = project_root_path / "data/inputs/test_questions.json"
    defaults["features_file_path"] = project_root_path / "data/features/test_questions.features.txt"
    defaults["predictions_file_path"] = project_root_path / "data/predictions/test_questions.predictions.txt"
    defaults["output_file_path"] = project_root_path / "data/outputs/test_questions.outputs.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type = str, default = defaults["input_file_path"])
    parser.add_argument("--features_file_path", type = str, default = defaults["features_file_path"])
    parser.add_argument("--predictions_file_path", type = str, default = defaults["predictions_file_path"])
    parser.add_argument("--output_file_path", type = str, default = defaults["output_file_path"])
    parser.add_argument("--process", action = "store_true", default = False)

    options = parser.parse_args(sys.argv[1:])

    generateSanitizedPredictions(options)
