from yaml import load
import logging

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
            print(repr(err))
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