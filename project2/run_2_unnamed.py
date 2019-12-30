from question_2 import *

log_file = os.path.join(os.getcwd(), 'question_2.log')
svm_decomposition = SVMDecomposition(logging_path=log_file)

num_points = len(svm_decomposition.train_y)
lambda_ = np.zeros((num_points, 1))
q = 100

svm_decomposition.optimize(lambda_, q)