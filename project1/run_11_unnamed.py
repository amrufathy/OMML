import numpy as np

from models import MLP, SEED, train_test_split

dataset = np.genfromtxt('data_points.csv', delimiter=',')

x = dataset[1:, :2]
y = dataset[1:, 2]
y = np.expand_dims(y, -1)  # row -> column vector

# train 70%, test 30%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=SEED)

mlp = MLP(hidden_size=10, _rho=1e-5)
mlp.fit(x_train, y_train)
print(f'Test error: {mlp.test_loss(x_test, y_test):.4f}')
mlp.surface_plot(x_test)
