import os

from question_3 import *

log_file = os.path.join(os.getcwd(), 'project2', 'question_3.log')
solver_mvp = SVMMVP(logging_path=log_file)

num_points = len(solver_mvp.train_y)
lambda_ = np.zeros((num_points, 1))

solver_mvp.optimize(lambda_)
