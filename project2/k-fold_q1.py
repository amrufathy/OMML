# # K-fold

k_folds = 5
x_train24, y_train24 = shuffle(x_train24, y_train24, random_state=SEED)
y_train24_sub = np.split(y_train24, k_folds)
x_train24_sub = np.split(x_train24, k_folds)

C_params = [0.01, 0.1, 1, 2, 2.5, 3, 6, 10, 100]
Gamma_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

result_dict = dict()
for c in C_params:
    for gamma in Gamma_params:
        param_key = str(c) + '__' + str(gamma)
        print(param_key)
        # one pair of parameters
        result_k_fold = []
        for k in range(k_folds):
            x_train, y_train, x_test, y_test = train_val(2, x_train24_sub, y_train24_sub)
            res = SVM_prediction(x_train, y_train, gamma, c, x_test, y_test)
            result_k_fold.append(res)
            acc = accuracy_score(y_test, res[1])
            if res[1] < 0.6:
                break
        result_dict.update({param_key: result_k_fold})

print(result_dict)
