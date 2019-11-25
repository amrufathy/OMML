import numpy as np

from models import MLP

dataset = np.genfromtxt('dataPointsTest.csv', delimiter=',')

x_test = dataset[1:, :2]
y = dataset[1:, 2]
y_test = np.expand_dims(y, -1)  # row -> column vector

mlp = MLP(hidden_size=25, _rho=1e-5)
mlp.load()

# test loss internally creates `y_pred` and then returns the
#   MSE between `y_pred` and `y_test`
print(f'MSE = {mlp.test_loss(x_test, y_test)}')
