from __future__ import absolute_import
import logging
import itertools
from typing import List, Dict, Optional, Union
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, \
    MultiLabelField, ListField

from src.reader.utils import get_sentence_markers_from_tokens

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    return empty_line


@DatasetReader.register("handcrafted_feature_reader")
class HandCraftedFeatureReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 features_index_map: Union[Dict[str, int], str],
                 feature_label_namespace: str = "feature_labels",
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 label_namespace: str = "labels",
                 use_sentence_markers: bool = False) -> None:
        super(HandCraftedFeatureReader, self).__init__(lazy=lazy)
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        if isinstance(features_index_map, str):
            with open(features_index_map, "r") as fil:
                _features_index_map: Dict[str, int] = {}
                for index, line in enumerate(fil):
                    line = line.strip()
                    assert line not in _features_index_map
                    _features_index_map[line] = index
            self._features_index_map = _features_index_map
        else:
            self._features_index_map = features_index_map
        self._coding_scheme = coding_scheme
        self.feature_label_namespace = feature_label_namespace
        self._use_sentence_markers = use_sentence_markers
        self._train = True

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    @overrides
    def _read(self, file_path: str) -> List[Instance]:
        instances: List[Instance] = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    tokens: List[str] = []
                    tags: List[str] = []
                    features: List[List[str]] = []
                    for field in fields:
                        tokens.append(field[0])
                        tags.append(field[-1])
                        features.append(field[1:-1])
                    tags = tags if self._train else None
                    instances.append(self.text_to_instance(
                        tokens=tokens, features=features, tags=tags))
        return instances

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         features: List[List[str]],
                         tags: Optional[List[str]] = None,
                         tag_label_namespace: Optional[str] = None):
        # pylint: disable=arguments-differ
        tokens: List[Token] = [Token(x) for x in tokens]
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        metadata = {"words": [x.text for x in tokens]}
        if self._use_sentence_markers:
            sentence_markers = get_sentence_markers_from_tokens(tokens)
            metadata["sentence_markers"] = sentence_markers
        instance_fields["metadata"] = MetadataField(metadata)
        # now encode the features
        feature_list: List[MultiLabelField] = []
        for feature in features:
            indexed_feature: List[int] = [
                self._features_index_map[x] for x in feature if x in self._features_index_map
            ]
            feature_list.append(MultiLabelField(indexed_feature, label_namespace=self.feature_label_namespace,
                                                skip_indexing=True, num_labels=len(self._features_index_map)))
        instance_fields["features"] = ListField(feature_list)
        if tags:
            tag_label_namespace = tag_label_namespace or self.label_namespace
            converted_tags: List[str] = self.convert_tags(tags)
            instance_fields["tags"] = SequenceLabelField(converted_tags,
                                                         sequence, tag_label_namespace)
        return Instance(instance_fields)

    @staticmethod
    def convert_tags(tags: List[str]) -> List[str]:
        """Converts tags into an IOB1 formatted tag structure
        """
        new_tags = []
        for tag in tags:
            new_tags.append(f"I-{tag}" if tag != "O" else tag)
        return new_tags

    def read_partial(self, file_path: str):
        instances: List[Instance] = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    tokens: List[str] = []
                    tags: List[str] = []
                    features: List[List[str]] = []
                    for field in fields:
                        tokens.append(field[0])
                        tags.append(field[-1])
                        features.append(field[1:-1])
                    instances.append(self.text_to_instance(
                        tokens=tokens, features=features, tags=tags,
                        tag_label_namespace="partial_labels"))
        return instances
