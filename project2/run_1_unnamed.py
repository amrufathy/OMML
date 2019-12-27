from question_1_utils import *
import os
import time
import numpy as np
from cvxopt import matrix, solvers
from data_extraction import load_mnist, SEED
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle

file = open('question_1.log', mode='w+')

data_path = os.path.join(os.getcwd(), 'Data')

x_train24, y_train24, x_test24, y_test24 = load_mnist(data_path, kind='train')
# best HPs:
gamma = 0.001  # for rbf kernel
c = 100

print(f'HYPERPARAMETERS', file=file)
print(f'GAMMA: {gamma}', file=file)
print(f'C: {c}', file=file)

res_train = SVM_prediction(x_train24, y_train24, gamma, c, x_train24, y_train24, False)
acc_train = accuracy_score(y_train24, res_train[1])
conf_matrix = confusion_matrix(y_train24, res_train[1])

print(f'TRAIN SET ACCURACY: {np.round(acc_train * 100, 2)}')

res_test = SVM_prediction(x_train24, y_train24, gamma, c, x_test24, y_test24, True)
acc_test = accuracy_score(y_test24, res_test[1])
conf_matrix = confusion_matrix(y_test24, res_test[1])  # first row tn, fp; second row fn, tp

print(f'TEST SET ACCURACY: { np.round(acc_test * 100,2)}', file=file)
print(f'TEST CONFUSION MATRIX: \n{conf_matrix}', file=file)
print(f"ITERATIONS:  {res_test[0]['iterations']}", file=file)
lambda_ = res_train[0]['x']
d = KKT_violation(np.array(lambda_),y_train24,c)
print(f'm - M: {d}', file=file)

file.close()