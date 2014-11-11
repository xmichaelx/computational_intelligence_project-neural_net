package com.sybillatechnologies.research;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Michał Drzał on 2014-11-11.
 */
public class NeuralNetwork {
    private SimpleMatrix nn_params;
    private int input_layer_size;
    private int hidden_layer_size;
    private int num_labels;

    public NeuralNetwork(int input_layer_size, int hidden_layer_size, int num_labels) {
        this.input_layer_size = input_layer_size;
        this.hidden_layer_size = hidden_layer_size;
        this.num_labels = num_labels;

        this.initializeRandomWeights();
    }

    private void initializeRandomWeights() {
        SimpleMatrix theta1 = Utils.randInitializeWeights(input_layer_size,hidden_layer_size);
        // theta2 is the set of weights corresponding to connections from hidden layer to output layer
        SimpleMatrix theta2 = Utils.randInitializeWeights(hidden_layer_size, num_labels);
        // unrolling them into a linearized matrix
        nn_params = Utils.unroll(theta1).combine(0, theta1.getNumElements(), Utils.unroll(theta2));
    }

    public void train(SimpleMatrix X, SimpleMatrix y, double lambda, int iterations) {
        CostFunction costFunction = new CostFunction(X,y,lambda,input_layer_size,hidden_layer_size,num_labels);
        // optimizing set of weights
        nn_params = Fmincg.minimizeFunction(costFunction,nn_params,iterations);
    }

    /**
     * Predict labels for given set of points and weights.
     * @param x set of data points
     * @return predicted labels for input data points
     */
    public SimpleMatrix predict(SimpleMatrix x) {
        SimpleMatrix theta1 = nn_params.extractMatrix(0,1,0,(input_layer_size+1)*hidden_layer_size);
        theta1.reshape(hidden_layer_size, input_layer_size + 1);

        SimpleMatrix theta2 = nn_params.extractMatrix(0,1,(input_layer_size+1)*hidden_layer_size,nn_params.getNumElements());
        theta2.reshape(num_labels, hidden_layer_size + 1);

        int m = x.numRows();
        SimpleMatrix ones = new SimpleMatrix(m, 1);
        ones.set(1.0);
        SimpleMatrix X = ones.combine(0,1,x);
        SimpleMatrix h1 = Utils.sigmoid(X.mult(theta1.transpose()));
        SimpleMatrix h2 = Utils.sigmoid(ones.combine(0, 1, h1).mult(theta2.transpose()));


        SimpleMatrix p = new SimpleMatrix(m,1);
        for (int i = 0;i<m;i++) {
            int maxj = 0;
            for (int j = 1;j < num_labels;j++) {
                if (h2.get(i, maxj) < h2.get(i, j)){
                    maxj = j;
                }
            }
            p.set(i,0,maxj+1);
        }

        return p;
    }

}
