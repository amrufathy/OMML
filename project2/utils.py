import logging


def initialize_logger():
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
