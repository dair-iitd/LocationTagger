from __future__ import absolute_import
from typing import List, Optional, NamedTuple, Dict, Tuple
import argparse
from copy import deepcopy
import os
import tqdm
import re
import torch
import random
import torch
import tqdm

from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import SequenceLabelField, TextField, MultiLabelField, Field, \
    ListField
from allennlp.data.iterators import DataIterator
from allennlp.data.dataset import Batch
from allennlp.common import Params
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.nn import util
from allennlp.common.util import JsonDict
from allennlp.training import util as training_util

from utils import read_from_config_file, bool_flag
from allennlp.common.util import sanitize


class ModelData(NamedTuple):
    metric: float
    model: torch.nn.Module = None
    weights: Dict[str, torch.Tensor] = None


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ccm/crf tagger")
    parser.add_argument("-cf", "--config_file", type=str, required=True,
                        help="The configuration file")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="The Output directory file")
    parser.add_argument("-d", "--devices", type=int,
                        default=[-1], nargs="+")

    parser.add_argument("-six", "--start_index", type=int, required=False, default=None)
    parser.add_argument("-eix", "--end_index", type=int, required=False, default=None)
    parser.add_argument("--save_model", type=bool_flag, required=False, default=False)
    args = parser.parse_args()
    return args


def get_trainer_from_config(config: Params,
                            train_instances: List[Instance],
                            val_instances: List[Instance],
                            vocab: Optional[Vocabulary] = None,
                            device: Optional[int] = -1) -> Trainer:
    trainer_params = config.pop("trainer")
    trainer_params["cuda_device"] = device
    model_params = config.pop("model")
    vocab = vocab or Vocabulary.from_instances(train_instances)
    model = Model.from_params(model_params, vocab=vocab)
    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(vocab)
    trainer = Trainer.from_params(
        model=model,
        iterator=iterator,
        train_data=train_instances,
        validation_data=val_instances,
        serialization_dir=None,
        params=trainer_params)
    return trainer


def post_process_prediction(prediction: JsonDict) -> List[str]:
    tags = prediction["tags"]
    tags = [re.sub(r"^.*-", "", tag) for tag in tags]
    return tags


def get_new_instance(instance: Instance, tags: List[str], reader: DatasetReader) -> Instance:
    # first copy over the tokens
    new_instance: Dict[str, Field] = {}
    tokens = instance.fields["tokens"].tokens
    sequence = TextField(tokens, reader._token_indexers)
    new_instance["tokens"] = sequence
    # now copy the tags
    new_instance["tags"] = SequenceLabelField(tags, sequence, reader.label_namespace)
    # now copy the handcrafted features
    feature_list: List[MultiLabelField] = []
    for feature in instance.fields["features"]:
        labels: List[int] = feature.labels
        feature_list.append(MultiLabelField(
            labels, label_namespace=reader.feature_label_namespace,
            skip_indexing=True, num_labels=len(reader._features_index_map))
        )
    new_instance["features"] = ListField(feature_list)
    return Instance(new_instance)


def get_training_data(partial_instances: List[Instance],
                      ccm_model: Model,
                      reader: DatasetReader) -> List[Instance]:
    partial_instances = deepcopy(partial_instances)
    positional_hard_constraints: List[List[Tuple[int, int]]] = []
    for instance in partial_instances:
        hard_constraints: List[Tuple[int, int]] = []
        token_to_index = ccm_model.vocab.get_token_to_index_vocabulary(ccm_model.label_namespace)
        for ix, label in enumerate(instance.fields["tags"].labels):
            if label in token_to_index:
                hard_constraints.append((ix, ccm_model.vocab.get_token_index(label, ccm_model.label_namespace)))
        positional_hard_constraints.append(hard_constraints)
    new_tags: List[List[str]] = []
    batch_size = 32
    for ix in tqdm.tqdm(range(0, len(partial_instances), batch_size)):
        batch_instances = partial_instances[ix: ix + batch_size]
        for instance in batch_instances:
            instance.fields.pop("tags", None)
        new_tags += ccm_model.forward_on_instances(
            batch_instances,
            positional_hard_constraints[ix: ix + batch_size])
    assert len(new_tags) == len(partial_instances)
    new_instance_list: List[Instance] = []
    for instance, tag in zip(partial_instances, new_tags):
        new_instance = get_new_instance(instance, tag, reader)
        new_instance_list.append(new_instance)
    return new_instance_list


def get_metrics(trainer: Trainer) -> Dict[str, float]:
    val_loss, batches_this_epoch = trainer._validation_loss()
    metrics = training_util.get_metrics(trainer.model, val_loss, batches_this_epoch)
    return metrics


def train_single(config: Params,
                 instances: List[Instance],
                 partially_labeled_instances: List[Instance],
                 reader: DatasetReader,
                 index: int,
                 cuda_device: int) -> List[str]:
    instances = deepcopy(instances)
    partially_labeled_instances = deepcopy(partially_labeled_instances)
    config = deepcopy(config)
    test_instance = instances[index]
    train_val_instances = instances[:index] + instances[index + 1:]

    random.shuffle(train_val_instances)
    num_train_instances = int(0.9 * len(train_val_instances))
    train_instances = train_val_instances[:num_train_instances]
    val_instances = train_val_instances[num_train_instances:]
    trainer = get_trainer_from_config(config.duplicate(),
                                      train_instances=train_instances,
                                      val_instances=val_instances,
                                      device=cuda_device)
    trainer.train()

    metric, should_decrease = trainer._validation_metric, trainer._metric_tracker._should_decrease
    patience = trainer._metric_tracker._patience
    bad_epochs = -1
    num_epochs = 5
    model = trainer.model
    metrics = get_metrics(trainer)
    originalModel = ModelData(metric=metrics[metric], weights=model.state_dict())
    bestModel = ModelData(metric=metrics[metric], model=model)

    for _ in range(num_epochs):
        best_model = bestModel.model
        train_instances = get_training_data(partially_labeled_instances, best_model, reader)
        trainer = get_trainer_from_config(
            config.duplicate(),
            train_instances=train_instances,
            val_instances=val_instances,
            vocab=best_model.vocab,
            device=cuda_device)
        trainer.train()
        # update the parameters
        with torch.no_grad():
            for name, value in trainer.model.named_parameters():
                if name in originalModel.weights and value.requires_grad:
                    value.mul_(0.1).add_(0.9 * originalModel.weights[name])
        metrics = get_metrics(trainer)
        if should_decrease:
            if metrics[metric] < bestModel.metric:
                bestModel = ModelData(metric=metrics[metric], model=trainer.model)
                bad_epochs = 0
            else:
                bad_epochs += 1
        else:
            if metrics[metric] > bestModel.metric:
                bestModel = ModelData(metric=metrics[metric], model=trainer.model)
                bad_epochs = 0
            else:
                bad_epochs += 1
        if bad_epochs == patience:
            break
    model = bestModel.model
    model.eval()
    prediction = sanitize(model.forward_on_instance(test_instance))
    return prediction, model


def serial_processing(instances: List[Instance],
                      partially_labeled_instances: List[Instance],
                      reader: DatasetReader,
                      config: Params, device: int,
                      serialization_dir: str, start_index: Optional[int] = None,
                      end_index: Optional[int] = None,
                      save_model: bool = False) -> None:
    start_index = start_index or 0
    end_index = end_index or len(instances)
    for index in tqdm.tqdm(range(start_index, end_index)):
        prediction, model = train_single(config, instances, partially_labeled_instances, reader, index, device)
        with open(os.path.join(serialization_dir, f"prediction_{index}.txt"), "w") as f:
            f.write("\n".join(prediction))
        if index == end_index - 1 and save_model:
            model_dir = os.path.join(serialization_dir, f"start_{start_index}_end_{end_index}")
            os.makedirs(model_dir, exists_ok=True)
            vocab = model.vocab
            vocab.save_to_files(os.path.join(model_dir, "vocabulary"))
            model_save_path = os.path.join(
                model_dir, f"best.th"
            )
            torch.save(model.state_dict(), model_save_path)


def main(args) -> None:
    config = read_from_config_file(args.config_file)
    serialization_dir = args.base_dir
    os.makedirs(serialization_dir, exist_ok=True)
    reader_params = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_params)
    data_file_path = config.pop("all_data_path")
    partially_labeled_data_path = config.pop("partial_data_path")
    instances = reader.read(data_file_path)
    partial_instances = reader.read_partial(partially_labeled_data_path)
    serial_processing(instances, partial_instances, reader, config, args.devices[0],
                      serialization_dir, args.start_index, args.end_index, args.save_model)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
