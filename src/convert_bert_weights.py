from __future__ import absolute_import
from typing import MutableMapping, Dict, Any
import torch
import argparse

from allennlp.nn.util import device_mapping


def get_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--new_path", type=str, required=True)
    args = parser.parse_args()
    return vars(args)


def convert_weights(state_dict: MutableMapping) -> MutableMapping:
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        if "LayerNorm" in key:
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


if __name__ == "__main__":
    args = get_arguments()
    old_weights = torch.load(args["path"], map_location=device_mapping(-1))
    new_weights = convert_weights(old_weights)
    torch.save(new_weights, args["new_path"])
