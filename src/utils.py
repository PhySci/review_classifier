from yaml import load
import logging
import torch
import random
import numpy as np

_logger = logging.getLogger(__name__)


def read_yml(pth: str) -> dict:
    """
    :param pth:
    :return:
    """
    res = None
    with open(pth, 'r') as fid:
        try:
            res = load(fid)
        except Exception as err:
            _logger.error(repr(err))
    return res


def setup_logging(logfile='log.txt', loglevel='INFO') -> None:
    """
    Sets up logger.

    :param logfile: a file name of the log file.
    :param loglevel: a level of logging to be used
    :return: None
    """
    loglevel = getattr(logging, loglevel)

    logger = logging.getLogger()
    logger.setLevel(loglevel)
    fmt = '%(asctime)s: %(levelname)s: %(filename)s: ' + \
          '%(funcName)s(): %(lineno)d: %(message)s'
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


def revert_listdict(d) -> dict:
    """
    Flatten and invert dictionary of lists
    {0: [a, b, c], 1: [d, e, f], ...} -> {a: 0, b: 0, c:0, d: 1, e: 1, f: 1, ...}

    :return: flatten and inverter dictionary
    """
    res = {}
    for label, cat_list in d.items():
        res.update({el: int(label) for el in cat_list})
    return res


def set_seeds(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    random.seed(seed+1)
    np.random.seed(seed+2)


