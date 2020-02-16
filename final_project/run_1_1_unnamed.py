from final_project.mlp import *

x_train57, y_train57, x_test57, y_test57 = load_binary_mnist('Data')
y_train57, y_test57 = y_train57.ravel(), y_test57.ravel()
y_train57[y_train57 == -1] = 0
y_test57[y_test57 == -1] = 0

mlp = MLP(input_size=x_train57.shape[1], hidden_size=100)

print(f'Number of neurons: {mlp.hidden_size}')
print(f'Initial training error: {mlp.test_loss(x_train57, y_train57)}')

train_time = mlp.fit(x_train57, y_train57)

print(f'Final training error: {mlp.test_loss(x_train57, y_train57):.4f}')
print(f'Final test error: {mlp.test_loss(x_test57, y_test57):.4f}')
print(f'Norm of the gradient at the final point: '
      f'{np.sum(grad_softmax_cross_entropy_with_logits(mlp.forward(x_train57), y_train57)):.4f}')
print(f'Optimization solver chosen: Batch Gradient Descent')
print(f'Total number of function evaluations: {mlp.nfev}')
print(f'Total number of gradient evaluations: {mlp.ngev}')
print(f'Train time: {mlp.train_time:.4f}')
print(f'Sigma: {1.0}')
print(f'Rho: {1e-04}')
