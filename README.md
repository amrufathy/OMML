# OMML

- Implementing NN, MLP, RBF from scratch using only `numpy` & `scipy`
- Data can be found at projet1/data_points.csv

## Implementation

- to install the requirements to run the Project, use `pip install -r requirements.txt`
- To run the code `python3 ommml/project1/run_xx_unnamed.py` the xx belongs to set of questions provided in the project {11, 12, 21, 22, 31, 32}
- Created [`models.py`](https://github.com/amrufathy/OMML/blob/master/project1/models.py), it contains definition of NeuralNetwork, MultiLayerPerceptron, RadialBasisFunctionNetwork
- [Multilayer Perceptron Training](https://github.com/amrufathy/OMML/blob/master/project1/run_11_unnamed.py)
- [Radial Basis Function Network Training](https://github.com/amrufathy/OMML/blob/master/project1/run_12_unnamed.py)

- [Multilayer Perceptron Extreme Learning](https://github.com/amrufathy/OMML/blob/master/project1/run_21_unnamed.py)
- [Radial Basis Function Extreme Learning](https://github.com/amrufathy/OMML/blob/master/project1/run_22_unnamed.py)

- [Radial Basis Function Network Decomposition](https://github.com/amrufathy/OMML/blob/master/project1/run_3_unnamed.py)


## Neural Network (NN)

- What is a Neural Network? it is an Artificial Network that tries to mimic the human's brain neurons, it is made up of 3 layers input layer where it receives the input and pass it to the 2nd layer which is the hidden layer, and the sum of weights as per number of neurons is added with the biases to give us our output.

- It is characterized by the following parameters `{input_size, hidden_size, output_size, rho}`


## MultiLayer Perceptron (MLP)

- It basically inherits the Neural Networks' parameters, but we have here the `{weights, biases}` and it uses learning techniques based on minimizing the Mean Squared Error between the True labels and the predicted labels by the model, it keeps on training till all samples are correctly classified or untill a certain stopping condition (threshold, num of epochs)

## Radial Basis Function Networks (RBF)

- It inherits the Neural Networks' parameters, just like the MLP, and it has another set of params {C, V}. It is similar to MLP however RBFs are based on radial basis function (RBF) -is a real-valued function- whose value depends only on the distance between the input and some fixed point.
