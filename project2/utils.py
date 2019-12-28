import logging
import random

import numpy as np


def initialize_logger(file):
    SEED = 1848399
    random.seed(SEED)
    np.random.seed(SEED)
    logging.basicConfig(filename=file,
                        format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
