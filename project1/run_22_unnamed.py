import numpy as np

from models import RBF, SEED, train_test_split

dataset = np.genfromtxt('data_points.csv', delimiter=',')

x = dataset[1:, :2]
y = dataset[1:, 2]
y = np.expand_dims(y, -1)  # row -> column vector

# train 70%, test 30%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=SEED)

rbf = RBF(hidden_size=10, _rho=1e-5, _sigma=1)
rbf.extreme_learning(x_train, y_train)
print(f'Test error: {rbf.test_loss(x_test, y_test):.4f}')
rbf.surface_plot(x_test)
