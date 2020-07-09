import sys
import json
import argparse
from utils import common
from collections import OrderedDict
from . import FeatureGenerator

def generateFeatureFileFromAnnotations(options):
    data = list(json.loads(open(options.data_file_path).read(), object_pairs_hook = OrderedDict).items())

    sentences = []
    labels = []
    for key, value in data:
        sentence, item = list(value.items())[0]
        ilabels = list(map(lambda x: x.split("|")[1].strip(), item))
        sentences.append(sentence)
        labels.append(ilabels)

    features = FeatureGenerator.getFeatures(sentences = sentences, labels = labels)
    with open(options.features_file_path, "w") as file:
        file.write(features)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["data_file_path"] = project_root_path / "data/inputs/test_annotations.json"
    defaults["features_file_path"] = project_root_path / "data/features/test_annotations.features.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type = str, default = defaults["data_file_path"])
    parser.add_argument("--features_file_path", type = str, default = defaults["features_file_path"])

    options = parser.parse_args(sys.argv[1:])

    generateFeatureFileFromAnnotations(options)
