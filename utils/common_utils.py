from __future__ import absolute_import
import argparse
from typing import List, Union, Optional
import logging
import sys
import os
from subprocess import Popen, PIPE
import json
import inspect
import io
import numpy as np
import spacy
import torch

from allennlp.common import Params
from allennlp.data import Token


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def bool_flag(s: str) -> bool:
    """
        Parse boolean arguments from the command line.
        ..note::
        Usage in argparse:
            parser.add_argument(
                "--cuda", type=bool_flag, default=True, help="Run on GPU")
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError(
        "invalid value for a boolean flag (0 or 1)")


def setup_logger(logfile: str = "", loglevel: str = "INFO") -> logging.RootLogger:
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=numeric_level, stream=sys.stdout)
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    if logfile != "":
        logfile_handle = logging.FileHandler(logfile, 'w')
        logfile_handle.setFormatter(fmt)
        logger.addHandler(logfile_handle)
    return logger


def setup_output_dir(config: Params, loglevel: Optional[str] = None) -> str:
    """Setup the Experiment Folder
    Note that the output_dir stores each run as run-1, ....
    Makes the next run directory. This also sets up the logger
    A run directory has the following structure
    - run-1
        - models
                * modelname*.tar.gz
                - vocabulary
                    * namespace_1.txt
                    * namespace_2.txt ...
        * config.json
        * githash.log of current run
        * gitdiff.log of current run
        * logfile.log (the log of the current run)
    Arguments:
        config (``allennlp.common.Params``): The experiment parameters
        loglevel (str): The logger mode [INFO/DEBUG/ERROR]
    Returns
        str, allennlp.common.Params: The filename, and the modified config
    """
    output_dir = config.get('base_output_dir', "./Outputs")
    make_directory(output_dir)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith('run-'):
            last_run = max(last_run, int(dirname.split('-')[1]))
    new_dirname = os.path.join(output_dir, 'run-%d' % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, 'models')
    make_directory(best_model_dirname)
    vocab_dirname = os.path.join(best_model_dirname, 'vocabulary')
    make_directory(vocab_dirname)

    config_file = os.path.join(new_dirname, 'config.jsonnet')
    write_config_to_file(config_file, config)

    # Save the git hash
    process = Popen(
        'git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    stdout = stdout.decode('ascii').strip('\n').strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)

    # Save the git diff
    process = Popen('git diff'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode('ascii', errors="ignore")
        fp.write(stdout)

    if loglevel:
        # Set up the logger
        logfile = os.path.join(new_dirname, 'logfile.log')
        setup_logger(logfile, loglevel)
        return best_model_dirname


def make_directory(dirname: str) -> None:
    """Constructs a directory with name dirname, if
    it doesn't exist. Can also take in a path, and recursively
    apply it.
    """
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        raise RuntimeError("Race condition. "
                           "Two methods trying to create to same place")


def read_from_config_file(filepath: str, params_overrides: str = "") -> Params:
    """Read Parameters from a config file
    Arguments:
        filepath (str): The file to read the
            config from
        params_overrides (str): Overriding the config
            Can potentially be used for command line args
            e.g. '{"model.embedding_dim": 10}'
    Returns:
        allennlp.common.Params: The parameters
    """
    return Params.from_file(
        params_file=filepath,
        params_overrides=params_overrides)


def write_config_to_file(filepath: str, config: Params) -> None:
    """Writes the config to a json file, specifed by filepath
    """
    with io.open(filepath, 'w', encoding='utf-8', errors='ignore') as fd:
        json.dump(fp=fd,
                  obj=config.as_dict(quiet=True),
                  ensure_ascii=False, indent=4, sort_keys=True)


def convert_spacy_token(token: spacy.tokens.token.Token) -> Token:  # pylint: disable=c-extension-no-member
        """We inspect the constructor of the AllenNLP Token to figure out how to
        populate the arguments
        Parameters
            token (`Spacy.Token`): The spacy generated token
        Returns:
            Token (`Allennlp.data.Token`): The AlleNLP generated token
        """
        kwargs = {}
        for arg in inspect.getfullargspec(Token.__init__).args:
            if arg == "self":
                continue
            if hasattr(token, f"{arg}_"):
                kwargs[arg] = getattr(token, f"{arg}_")
            elif hasattr(token, arg):
                kwargs[arg] = getattr(token, arg)
        return Token(**kwargs)


def convert_spacy_token_list(
    tokens: List[Union[spacy.tokens.token.Token, Token]]  # pylint: disable=c-extension-no-member
) -> List[Token]:
    """Working with spacy tokens is hard, since they don't allow
    for easy manipulation. Is easier to work with allennlp tokens
    Note that this performs an *in place* conversion
    """
    for ix, _ in enumerate(range(len(tokens))):
        if isinstance(tokens[ix], Token):
            continue
        else:
            # this is a spacy token
            tokens[ix] = convert_spacy_token(tokens[ix])
    return tokens


def convert_sparse_to_dense(matrix: torch.Tensor, offsets: torch.LongTensor) -> torch.Tensor:
    """Converts a sparse matrix of bs x seq_len x d1 ... into a dense num_non_zero x d1 ...
    We assume the zeroed out entries are right padded.

    Parameters:
        matrix (``torch.Tensor``): bs x seq_len x d1 ...: the matrix to densify
        offsets (``torch.LongTensor``): bs, : the offset, where
            bs[i + 1] - bs[i] == number of non zero entries
        for ith entry
    Returns:
        dense_mat (``torch.Tensor``): num_non_zero x d1 ..
    """
    dense_mat = torch.zeros(offsets[-1], *matrix.size()[2:], out=torch.empty_like(matrix))
    prev_ix = 0
    for bs, next_ix in enumerate(offsets):
        dense_mat[prev_ix:next_ix] = matrix[bs, :(next_ix - prev_ix)]
        prev_ix = next_ix
    return dense_mat


def convert_dense_to_sparse(dense_mat: torch.Tensor,
                            offsets: torch.LongTensor,
                            max_size: Optional[int] = None) -> torch.Tensor:
    """Converts a dense matrix of num_non_zero, ... into a sparse matrix of shape bs x max_size x ...
    The offset array defines the value of the non zero entries

    Parameters:
        dense_mat (``torch.Tensor``): num_non_zero x d1 ...
        offsets (``torch.Tensor``): batch: The offsets
        max_size (int): The maximum size of the sparse matrix. If not specified,
            we fit the tightest fit (i.e the maximum number of entries for a batch)

    Returns:
        sparse_mat (``torch.Tensor``): batch x max_size x d1 ...
    """
    if max_size is None:
        max_size = offsets[0].item() if offsets.size(0) == 1 else max(
            (offsets[1:] - offsets[:-1]).max().item(),
            offsets[0].item()
        )
    sparse_mat = torch.zeros(offsets.size(0), max_size, *dense_mat.size()[1:], out=torch.empty_like(dense_mat))
    prev_ix = 0
    for bs, next_ix in enumerate(offsets):
        sparse_mat[bs, :(next_ix - prev_ix)] = dense_mat[prev_ix: next_ix]
        prev_ix = next_ix
    return sparse_mat
