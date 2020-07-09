from __future__ import absolute_import
from typing import List, Union

from allennlp.data import Token


def get_text(token: Union[str, Token]) -> str:
    if isinstance(token, str):
        return token
    return token.text


def get_sentence_markers_from_tokens(tokens: List[Union[str, Token]]) -> List[int]:
    punctuation_set = set([".", "?", "!"])
    sentence_markers: List[int] = []
    for ix in range(1, len(tokens)):
        token = tokens[ix]
        if get_text(token) in punctuation_set:
            continue
        if get_text(tokens[ix - 1]) in punctuation_set:
            sentence_markers.append(ix)
    sentence_markers.append(len(tokens))
    return sentence_markers


def get_sentences_from_markers(tokens: List[Union[str, Token]], markers: List[int]) -> List[List[Union[str, Token]]]:
    sentences: List[List[Token]] = []
    start = 0
    for end in markers:
        sentences.append(tokens[start: end])
        start = end
    return sentences
