import logging


def pipeline():
    pipeline_logger = logging.getLogger(__name__)
    pipeline_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:')

    pipeline_file_handler = logging.FileHandler('../logs/pipeline.log')
    pipeline_file_handler.setLevel(logging.DEBUG)
    pipeline_file_handler.setFormatter(formatter)

    pipeline_logger.addHandler(pipeline_file_handler)
    return pipeline_logger


def frontend():
    frontend_logger = logging.getLogger(__name__)
    frontend_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:')

    frontend_file_handler = logging.FileHandler('../logs/frontend.log')
    frontend_file_handler.setLevel(logging.DEBUG)
    frontend_file_handler.setFormatter(formatter)

    frontend_logger.addHandler(frontend_file_handler)
    return frontend_logger


def backend():
    backend_logger = logging.getLogger(__name__)
    backend_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:')

    backend_file_handler = logging.FileHandler('../logs/backend.log')
    backend_file_handler.setLevel(logging.DEBUG)
    backend_file_handler.setFormatter(formatter)

    backend_logger.addHandler(backend_file_handler)
    return backend_logger
