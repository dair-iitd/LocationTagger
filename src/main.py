from __future__ import absolute_import

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import tqdm
import torch
import random
import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.common import Params
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.common.util import JsonDict
from allennlp.nn import util

from utils import read_from_config_file, bool_flag
from allennlp.common.util import sanitize

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("BERT-CRF Tagger")

    parser.add_argument("--config_file", type = str, required = True, help = "Configuration File")
    parser.add_argument("--devices", type = int, default = [-1], nargs = "+")

    parser.add_argument("--data_file_path", type = str, required = True, help = "Train or Test File")
    parser.add_argument("--pretrained_model_path", type = str, required = False, default = None, help = "Pre-trained Model Path")

    parser.add_argument("--train", action = "store_true", default = False)
    parser.add_argument("--num_epochs", type = int, required = False, default = None, help = "Number of epochs")
    parser.add_argument("--serialization_dir", type = str, required = False, default = None, help = "Directory for Model Dumps")

    parser.add_argument("--test", action = "store_true", default = False)
    parser.add_argument("--predictions_file_path", type = str, required = False, default = None, help = "Prediction File Path")

    args = parser.parse_args()
    return args

def get_trainer_from_config(config: Params, train_instances: List[Instance], validation_instances: List[Instance], device: int, serialization_dir: str, model = None, pretrained_model_path = None) -> Trainer:
    trainer_params = config.pop("trainer")
    trainer_params["cuda_device"] = device

    model_params = config.pop("model")

    vocab_dir = config.pop("vocab_dir", None)
    vocab = Vocabulary.from_instances(train_instances) if(vocab_dir is None) else Vocabulary.from_files(vocab_dir)

    if(model is None):
        model = Model.from_params(model_params, vocab = vocab)
    if(pretrained_model_path is not None):
        model.load_state_dict(torch.load(pretrained_model_path, map_location = util.device_mapping(device)))

    iterator = DataIterator.from_params(config.pop("iterator"))
    trainer_params["num_serialized_models_to_keep"] = 1
    iterator.index_with(vocab)

    trainer = Trainer.from_params(model = model, iterator = iterator, train_data = train_instances, validation_data = validation_instances, serialization_dir = serialization_dir, params = trainer_params)
    return trainer

def train(config: Params, instances: List[Instance], index: int, device: int, num_epochs: int, serialization_dir: str, pretrained_model_path: Optional[str] = None):
    config = deepcopy(config)
    num_train_instances = int(0.85 * len(instances))

    for epoch in range(num_epochs):
        print("Epoch:", epoch)

        instancescopy = deepcopy(instances)
        train_instances = instancescopy[:num_train_instances]
        validation_instances = instancescopy[num_train_instances:]

        random.shuffle(train_instances)

        if(epoch == 0):
            trainer = get_trainer_from_config(config = deepcopy(config), train_instances = train_instances, validation_instances = validation_instances, device = device, serialization_dir = os.path.join(serialization_dir, ("epoch%d" % epoch)), model = None, pretrained_model_path = pretrained_model_path)
        else:
            trainer = get_trainer_from_config(config = deepcopy(config), train_instances = train_instances, validation_instances = validation_instances, device = device, serialization_dir = os.path.join(serialization_dir, ("epoch%d" % epoch)), model = model, pretrained_model_path = None)

        trainer.train()
        model = trainer.model

        model.eval()
        torch.save(model.state_dict(), os.path.join(serialization_dir, ("model_%d.weights" % epoch)))
        model.train()

def test(config: Params, instances: List[Instance], device: int, predictions_file_path: str, pretrained_model_path: str):
    trainer = get_trainer_from_config(config = deepcopy(config), train_instances = instances, validation_instances = instances, device = device, serialization_dir = ".", model = None, pretrained_model_path = pretrained_model_path)
    model = trainer.model
    model.eval()

    predictions = []
    bar = tqdm.tqdm(total = len(instances))
    for instance in instances:
        tags = sanitize(model.forward_on_instance(instance))
        predictions.append([tag.split("-", 1)[1] if "-" in tag else tag for tag in tags])
        sys.stdout = open(os.devnull, "w")
        bar.update()
        sys.stdout = sys.__stdout__

    data = "\n\n".join(["\n".join(prediction) for prediction in predictions])

    os.makedirs(Path(predictions_file_path).parent, exist_ok = True)
    file = open(predictions_file_path, "w")
    file.write(data)
    file.close()

    sys.stdout = sys.__stdout__

def main(args) -> None:
    if(not (args.train ^ args.test)):
        raise Exception("Please select only one of train and test modes")

    if(args.train):
        if(args.serialization_dir is None):
            raise Exception("Serialization Dir should be specified in train mode")

    if(args.test):
        if(args.pretrained_model_path is None):
            raise Exception("Pre-trained model path should be specified in test mode")
        if(args.predictions_file_path is None):
            raise Exception("Predictions file path should be specified in test mode")

    sys.stdout = open(os.devnull, "w")

    config = read_from_config_file(args.config_file)
    os.makedirs(args.serialization_dir, exist_ok = True)
    reader_params = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_params)
    instances = reader.read(args.data_file_path)

    if(args.train):
        train(config = config, instances = instances, device = args.devices[0], num_epochs = args.num_epochs, serialization_dir = args.serialization_dir, pretrained_model_path = args.pretrained_model_path)
    if(args.test):
        test(config = config, instances = instances, device = args.devices[0], predictions_file_path = args.predictions_file_path, pretrained_model_path = args.pretrained_model_path)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
