import os
import re
import sys
import nltk
import argparse
import itertools
import numpy as np
from typing import List

class Post(object):
    def __init__(self):
        self.sentence = None
        self.gold_instances = None
        self.prediction_instances = None
        self.precision = None
        self.recall = None

    @staticmethod
    def lcs(s: str, t: str) -> int:
        s = s.split(" ")
        t = t.split(" ")

        n = len(s)
        m = len(t)

        dp = [[0] * (m + 1)] * (n + 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if(s[i - 1] == t[j - 1]):
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[n][m]

    @staticmethod
    def getInstances(tokens: List[str], pos_tags: List[str], iob_tags: List[str]) -> "List[str]":
        conlltags = [(token, pos_tag, iob_tag) for token, pos_tag, iob_tag in zip(tokens, pos_tags, iob_tags)]
        tree = nltk.chunk.conlltags2tree(conlltags)

        instances = set()
        for subtree in tree:
            if(type(subtree) == nltk.Tree):
                instance = " ".join([token for token, pos_tag in subtree.leaves()])
                instances.add(instance)

        return list(instances)

    @staticmethod
    def getMultiMatchPrecision(gold_instances: List[str], prediction_instances: List[str]):
        m = len(gold_instances)
        n = len(prediction_instances)

        if(n == 0):
            return []

        if(m == 0):
            return [(0., prediction_instances[i], "") for i in range(n)]

        l = []
        for i in range(n):
            max_match = -1
            match_j = -1

            for j in range(m):
                lcs = Post.lcs(gold_instances[j], prediction_instances[i])
                if(max_match < lcs):
                    max_match = lcs
                    match_j = j

            l.append((float(max_match) / len(prediction_instances[i].split(" ")), prediction_instances[i], gold_instances[match_j]))

        return l

    @staticmethod
    def getMultiMatchRecall(gold_instances: List[str], prediction_instances: List[str]):
        m = len(gold_instances)
        n = len(prediction_instances)

        if(m == 0):
            return []

        if(n == 0):
            return [(0., "", gold_instances[i]) for i in range(m)]

        l = []
        for i in range(m):
            max_match = 0
            match_j = -1

            for j in range(n):
                lcs = Post.lcs(gold_instances[i], prediction_instances[j])
                if(max_match < lcs):
                    max_match = lcs
                    match_j = j

            l.append((float(max_match) / len(gold_instances[i].split(" ")), prediction_instances[match_j], gold_instances[i]))

        return l

    def setMetrics(self, tokens: List[str], pos_tags: List[str], gold: List[str], prediction: List[str]):
        self.sentence = " ".join(tokens)
        assert(len(tokens) == len(pos_tags) == len(gold) == len(prediction))

        self.gold_instances = Post.getInstances(tokens, pos_tags, gold)
        self.prediction_instances = Post.getInstances(tokens, pos_tags, prediction)

        self.precisions = self.getMultiMatchPrecision(self.gold_instances, self.prediction_instances)
        self.recalls = self.getMultiMatchRecall(self.gold_instances, self.prediction_instances)

    def __str__(self):
        buffer = "POST\n"
        buffer += self.sentence + "\n"
        buffer += "Golds: %s\n" % (str(self.gold_instances))
        buffer += "Predicitons: %s\n" % (str(self.prediction_instances))
        buffer += "Match Precisions: %s\n" % (str(self.precisions))
        buffer += "Match Recalls: %s\n" % (str(self.recalls))
        return buffer

class Evaluator(object):
    def __init__(self, posts: List[Post]) -> None:
        self.posts = posts

    @staticmethod
    def getDataFromFile(file_path: str) -> List[List[str]]:
        data: List[List[str]] = []
        with open(file_path, "r") as file:
            for is_divider, element in itertools.groupby(file.readlines(), lambda line: line.strip() == ""):
                if(not is_divider):
                    item: List[str] = [line.strip() for line in element]
                    data.append(item)
        return data

    @classmethod
    def create(cls, gold_file_path: str, prediction_file_path: str) -> "Evaluator":
        posts = Evaluator.getDataFromFile(gold_file_path)
        predictions = Evaluator.getDataFromFile(prediction_file_path)

        assert len(posts) == len(predictions)
        assert all([len(post) == len(prediction) for post, prediction in zip(posts, predictions)])

        post_objects = []
        for post, prediction in zip(posts, predictions):
            tokens = [token_features.split(" ")[0] for token_features in post]
            pos_tags = [pos_tag for token, pos_tag in nltk.pos_tag(tokens)]
            gold = [token_features.split(" ")[-1] for token_features in post]

            post_object = Post()
            post_object.setMetrics(tokens, pos_tags, gold, prediction)

            post_objects.append(post_object)

        return cls(post_objects)

    @staticmethod
    def computeMicro(data: List[List[float]]) -> float:
        return np.average(list(map(np.average, data)))

    @staticmethod
    def computeMacro(data: List[List[float]]) -> float:
        return np.average(list(itertools.chain.from_iterable(data)))

    def computeAggregateMetrics(self):
        precisions = list(filter(lambda x: x != [], [[item[0] for item in post.precisions] for post in self.posts]))
        recalls = list(filter(lambda x: x != [], [[item[0] for item in post.recalls] for post in self.posts]))

        self.micro_precision = Evaluator.computeMicro(precisions) if precisions else 0.0
        self.micro_recall = Evaluator.computeMicro(recalls) if recalls else 0.0
        self.macro_precision = Evaluator.computeMacro(precisions) if precisions else 0.0
        self.macro_recall = Evaluator.computeMacro(recalls) if recalls else 0.0

    def __str__(self):
        buffer = ""

        buffer += "1. MICRO\n"

        buffer += "- PRECISION : %.4f\n" % self.micro_precision
        buffer += "- RECALL    : %.4f\n" % self.micro_recall
        buffer += "- F1-SCORE  : %.4f\n" % (2 * self.micro_precision * self.micro_recall / (self.micro_precision + self.micro_recall + 1e-6))

        buffer += "\n2. MACRO\n"

        buffer += "- PRECISION : %.4f\n" % self.macro_precision
        buffer += "- RECALL    : %.4f\n" % self.macro_recall
        buffer += "- F1-SCORE  : %.4f\n" % (2 * self.macro_precision * self.macro_recall / (self.macro_precision + self.macro_recall + 1e-6))

        return buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Evaluator Function")
    parser.add_argument("--prediction_file_path", type = str, help = "Prediction File Path", required = True)
    parser.add_argument("--gold_file_path", type = str, help = "Gold File Path", required = True)
    options = parser.parse_args(sys.argv[1:])

    evaluator = Evaluator.create(options.gold_file_path, options.prediction_file_path)
    evaluator.computeAggregateMetrics()

    print(str(evaluator))
