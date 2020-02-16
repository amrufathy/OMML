from final_project.data_extraction import load_multi_class_mnist
from final_project.svm import MultiClassSVM

data_path = 'Data'
x_train579, y_train579, x_test579, y_test579 = load_multi_class_mnist(data_path)

svm = MultiClassSVM()
train_time = svm.fit(x_train579, y_train579)

print('Multi class strategy: One-vs-One')
print('Optimization solver: MVP Decomposition for each SVM')
print(f'Misclassified rate (train): {1 - svm.evaluate(x_train579, y_train579):.4f}')
print(f'Misclassified rate (test): {1 - svm.evaluate(x_test579, y_test579):.4f}')
print(f'Training time: {train_time:.4f} sec')
