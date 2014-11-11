package com.sybillatechnologies.research;
import org.ejml.ops.CommonOps;
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
        return CostFunction.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels ,X, y, lambda);
    }

    /**
     * Cost function representing error on labeled training data set for given weights.
     * @param nn_params linearized connection weights
     * @param input_layer_size size of input layer
     * @param hidden_layer_size size of hidden layer size
     * @param num_labels number of labels
     * @param x feature matrix
     * @param y labels matrix
     * @param lambda regularization parameter
     * @return tuple containing cost for current set of weights and error gradient for each connection weights
     */
    private static CostGradientTuple nnCostFunction(SimpleMatrix nn_params, int input_layer_size, int hidden_layer_size, int num_labels,
                                                   SimpleMatrix x, SimpleMatrix y, double lambda) {

        SimpleMatrix theta1 = nn_params.extractMatrix(0,1,0,(input_layer_size+1)*hidden_layer_size);
        theta1.reshape(hidden_layer_size, input_layer_size + 1);

        SimpleMatrix theta2 = nn_params.extractMatrix(0,1,(input_layer_size+1)*hidden_layer_size,nn_params.getNumElements());
        theta2.reshape(num_labels, hidden_layer_size + 1);

        // Setup some useful variables
        int m = x.numRows();
        // Add ones to the X data matrix
        // append bias terms
        // matlab: X = [ones(m, 1) X];
        SimpleMatrix ones = new SimpleMatrix(m, 1);
        ones.set(1.0);
        SimpleMatrix X = ones.combine(0,1,x);

        int max = Math.round((float)y.elementMaxAbs());
        SimpleMatrix Y = new SimpleMatrix(y.numRows(), max);
        for (int i = 0;i<y.numRows();i++) {
            int val = Math.round((float)y.get(i,0)) - 1;
            Y.set(i,val,1.0);
        }

        // matlab: Map from Layer 1 to Layer 2
        // matlab: z2=X*Theta1';
        SimpleMatrix z2 = X.mult(theta1.transpose());

        // matlab: z2=sigmoid(z2);
        // append bias terms
        // a2=[ones(m, 1) a2];
        SimpleMatrix a2 = ones.combine(0,1,Utils.sigmoid(z2));

        // matlab: z3=a2*theta2';
        SimpleMatrix z3 = a2.mult(theta2.transpose());
        // matlab: a3=sigmoid(z3);
        SimpleMatrix a3 = Utils.sigmoid(z3);

        // Compute cost
        // logisf=(-y).*log(z3)-(1-y).*log(1-z3);
        SimpleMatrix logisf = Y.negative().elementMult(a3.elementLog());
        SimpleMatrix second = Y.negative().plus(1.0).elementMult(a3.negative().plus(1.0).elementLog());
        CommonOps.subtract(logisf.getMatrix(), second.getMatrix(), logisf.getMatrix());

        // remove bias terms from theta
        SimpleMatrix theta1s = theta1.extractMatrix(0,theta1.numRows(),1,theta1.numCols());
        SimpleMatrix theta2s = theta2.extractMatrix(0,theta2.numRows(),1,theta2.numCols());

        double J = (((1.0 / m) * logisf.elementSum()))  + (lambda / (2.0 * m))
                * (theta1s.elementPower(2.0).elementSum() + theta2s.elementPower(2.0).elementSum());

        SimpleMatrix delta_3 = a3.minus(Y);
        z2 = ones.combine(0,1,z2);
        SimpleMatrix delta_2 = delta_3.mult(theta2).elementMult(Utils.sigmoidGradient(z2));
        delta_2 = delta_2.extractMatrix(0,delta_2.numRows(), 1, delta_2.numCols());
        SimpleMatrix theta1_grad = delta_2.transpose().mult(X).scale(1.0/m);
        SimpleMatrix theta2_grad = delta_3.transpose().mult(a2).scale(1.0/m);

        SimpleMatrix grad = Utils.unroll(theta1_grad).combine(0,theta1_grad.getNumElements(), Utils.unroll(theta2_grad));

        return new CostGradientTuple(grad,J);
    }
}
