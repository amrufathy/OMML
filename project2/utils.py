import logging
import random

import numpy as np
from cvxopt import solvers


def initialize_logger(file):
    SEED = 1848399
    random.seed(SEED)
    np.random.seed(SEED)

    solvers.options['show_progress'] = False

    logging.basicConfig(filename=file,
                        format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
