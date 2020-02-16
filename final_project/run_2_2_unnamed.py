from final_project.data_extraction import load_binary_mnist
from final_project.svm import SVM

data_path = 'Data'

x_train57, y_train57, x_test57, y_test57 = load_binary_mnist(data_path)

svm = SVM()
train_time, iters = svm.decomposition(x_train57, y_train57, max_iters=2000)

print(f'Misclassified rate (train): {1 - svm.evaluate(x_train57, y_train57):.4f}')
print(f'Misclassified rate (test): {1 - svm.evaluate(x_test57, y_test57):.4f}')
print(f'Training time: {train_time:.4f} sec')
print(f'C = {svm.C}, gamma = {svm.gamma}')

print(f'Iterations: {iters}')
