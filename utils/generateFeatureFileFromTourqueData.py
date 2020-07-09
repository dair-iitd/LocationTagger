import sys
import json
import argparse
from utils import common
from . import FeatureGenerator

def generateFeatureFileFromTourqueData(options):
    data = json.load(open(options.data_file_path, "r"))
    sentences = sorted(list({item["question"] for item in data}))
    features = FeatureGenerator.getFeatures(sentences = sentences)
    with open(options.features_file_path, "w") as file:
        file.write(features)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["data_file_path"] = project_root_path / "data/inputs/test_questions.json"
    defaults["features_file_path"] = project_root_path / "data/features/test_questions.features.2.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type = str, default = defaults["data_file_path"])
    parser.add_argument("--features_file_path", type = str, default = defaults["features_file_path"])

    options = parser.parse_args(sys.argv[1:])

    generateFeatureFileFromTourqueData(options)
