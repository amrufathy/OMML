import itertools
from question_2 import *
from sklearn.model_selection import KFold


SEED = 1873337

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


C_params = [0.01, 0.1, 1, 2, 2.5, 3, 6, 10, 100]
Gamma_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
combinations = list(itertools.product(C_params, Gamma_params))

res_dict = dict()
for combination in combinations:
    log_file = os.path.join(
        os.getcwd(), f'question_2_C_{combination[0]}_G_{combination[1]}.log')
    svm_decomposition = SVMDecomposition(
        logging_path=log_file, C=combination[0], gamma=combination[1])
    train_x = svm_decomposition.train_x
    train_y = svm_decomposition.train_y
    test_x = svm_decomposition.test_x
    test_y = svm_decomposition.test_y

    num_points = len(svm_decomposition.train_y)
    lambda_ = np.zeros((num_points, 1))
    q = 100

    # one pair of parameters
    train_acc, test_acc = [], []
    kf_iterations, kf_computational_time = [], []
    dual_objects = []

    for train_index, val_index in kf.split(train_x):
        svm_decomposition.train_x = train_x[train_index]
        svm_decomposition.test_x = train_x[val_index]

        svm_decomposition.train_y = train_y[train_index]
        svm_decomposition.test_y = train_y[val_index]

        svm_decomposition.optimize(lambda_, q)

        y_pred_train = svm_decomposition.predict(svm_decomposition.train_x)
        y_pred_val = svm_decomposition.predict(svm_decomposition.test_x)

        train_acc.append(svm_decomposition.acc(
            svm_decomposition.train_x, svm_decomposition.train_y))
        test_acc.append(svm_decomposition.acc(
            svm_decomposition.test_x, svm_decomposition.test_y))
        kf_iterations.append(svm_decomposition.iterations)
        kf_computational_time.append(svm_decomposition.computational_time)
        dual_objects.append(svm_decomposition.dual_obj)

    tot_train_acc_mean = np.mean(train_acc)
    tot_test_acc_mean = np.mean(test_acc)
    comp_time = np.mean(kf_computational_time)
    dual_obj = np.mean(dual_objects)
    iters = int(np.mean(kf_iterations))

    res_dict.update({combination: [svm_decomposition.C, svm_decomposition.gamma,
                                   dual_obj, iters, comp_time,
                                   tot_train_acc_mean, tot_test_acc_mean]})
