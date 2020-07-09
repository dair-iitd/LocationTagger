from __future__ import absolute_import
import logging
import itertools
from typing import List, Dict, Optional
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    return empty_line


@DatasetReader.register("feature_reader")
class FeatureReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 label_namespace: str = "labels") -> None:
        super(FeatureReader, self).__init__(lazy=lazy)
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        self._coding_scheme = coding_scheme

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
                    for field in fields:
                        tokens.append(field[0])
                        tags.append(field[-1])
                    instances.append(self.text_to_instance(tokens=tokens, tags=tags))
        return instances

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         tags: Optional[List[str]] = None):
        # pylint: disable=arguments-differ
        tokens: List[Token] = [Token(x) for x in tokens]
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags:
            converted_tags: List[str] = self.convert_tags(tags)
            instance_fields["tags"] = SequenceLabelField(converted_tags,
                                                         sequence, self.label_namespace)
        return Instance(instance_fields)

    @staticmethod
    def convert_tags(tags: List[str]) -> List[str]:
        """Converts tags into an IOB1 formatted tag structure
        """
        new_tags = []
        for tag in tags:
            new_tags.append(f"I-{tag}" if tag != "O" else tag)
        return new_tags
