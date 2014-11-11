package com.sybillatechnologies.research;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by Michał Drzał on 2014-11-09.
 * based on Matlab Code from one of assignments from Machine Learning Course by Andrew Ng
 */
public class CostFunction {
    private int input_layer_size;
    private int hidden_layer_size;
    private int num_labels;
    private SimpleMatrix X;
    private SimpleMatrix y;
    private double lambda;

    public CostFunction(SimpleMatrix x, SimpleMatrix y, double lambda,
                        int input_layer_size, int hidden_layer_size, int num_labels) {
        this.X = x;
        this.y = y;
        this.lambda = lambda;
        this.input_layer_size = input_layer_size;
        this.hidden_layer_size = hidden_layer_size;
        this.num_labels = num_labels;
    }

    public CostGradientTuple getCost(SimpleMatrix nn_params) {
        return Utils.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels ,X, y, lambda);
    }
}
