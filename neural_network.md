Neural network
==============

ANN implemented during project is a standard feedforward neural net with one hidden layer. 

![Network geometry](http://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

Thus it has two weight matrices for mapping layers: input to hidden and hidden to output. 
They are linearized into matrix θ.

Cost function is defined as follows:

![cost function](cost_function.png)

Where:

- m is number of examples in training set
- λ is regularization parameter (penalizing for high weights)
- θ represents weighs, upper superscript represents layer number,
 lower script coordinates represent connection between j-th neuron in current layer and i-th neuron in next layer
- x is a matrix containing training data, one example per row - x superscript i represents i-th example in the set
- y is a matrix of labels - y superscript i represents i-th example in the set
- h subscript θ represents hypothesis about the data corresponding to weights θ

Cost function is then optimized using nonlinear conjugate gradient method [wikipedia](http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)

Neural network code is an Java adaptation Exercise 4 from Coursera course on Machine Learning by Andrew Ng. 
Assignments are available in another GitHub [repository](https://github.com/yhyap/machine-learning-coursera).

Course available (after registration) on [Coursera](https://www.coursera.org/course/ml).




