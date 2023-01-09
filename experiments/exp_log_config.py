import logging


def logger_exp():
    exp_logger = logging.getLogger(__name__)
    exp_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:')

    backend_file_handler = logging.FileHandler('exp_run.log')
    backend_file_handler.setLevel(logging.DEBUG)
    backend_file_handler.setFormatter(formatter)

    exp_logger.addHandler(backend_file_handler)

    return exp_logger


exp = logger_exp()
