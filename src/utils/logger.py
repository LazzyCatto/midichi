import logging
import sys
import warnings

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s (%(name)s) [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) 

    console_formatter = logging.Formatter("(%(name)s) %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    sys.excepthook = log_uncaught_exceptions

    warnings.simplefilter("always")
    logging.captureWarnings(True)

    return logger


def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    logger = logging.getLogger()
    logger.error("Необработанное исключение", exc_info=(exc_type, exc_value, exc_tb))


def get_logger():
    return logging.getLogger()
