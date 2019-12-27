import logging
import random

import numpy as np


def initialize_logger():
    SEED = 1848399
    random.seed(SEED)
    np.random.seed(SEED)
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
