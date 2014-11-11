package com.sybillatechnologies.research;

import net.lingala.zip4j.core.ZipFile;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import sun.java2d.pipe.SpanShapeRenderer;

import java.io.FileOutputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.StringTokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Created by Michał Drzał on 2014-11-09.
 * based on Matlab Code from one of assignments from Machine Learning Course by Andrew Ng
 */
public class Utils {
    /**
     * Computes element-wise sigmoid function \sigma (x) = \frac{1}{1 + e^{-x}} and returns as a matrix.
     * @param a Input matrix
     * @return Matrix containing elements in form b_{ij} = \frac{1}{1 + e^{-a_{ij}}}
     */
    public static SimpleMatrix sigmoid(SimpleMatrix a) {
        SimpleMatrix b = new SimpleMatrix(a);
        CommonOps.changeSign(b.getMatrix());
        CommonOps.elementExp(b.getMatrix(), b.getMatrix());
        CommonOps.add(b.getMatrix(), 1.0);
        CommonOps.elementPower(b.getMatrix(), -1, b.getMatrix());

        return b;
    }

    /**
     * Computes element-wise sigmoid derivate: \sigma (x)(1 - \sigma(x)) and returns as a matrix.
     * @param a Input matrix
     * @return Matrix containing elements in form b_{ij} = \sigma (a_{ij})(1 - \sigma(a_{ij}))
     */
    public static SimpleMatrix sigmoidGradient(SimpleMatrix a) {
        SimpleMatrix sigZ = sigmoid(a);
        SimpleMatrix b = new SimpleMatrix(sigZ);
        CommonOps.changeSign(b.getMatrix());
        CommonOps.add(b.getMatrix(), 1.0);
        CommonOps.elementMult(b.getMatrix(), sigZ.getMatrix());

        return b;
    }

    /**
     * Unrolls matrix into a vector.
     * @param a matrix
     * @return linearized representation of input matrix
     */
    public static SimpleMatrix unroll(SimpleMatrix a) {
        SimpleMatrix b = a.copy();
        b.reshape(1, b.getNumElements());
        return b;
    }

    /**
     * Initializes random weights for connections between two layers of neural network.
     * @param incomingConnections
     * @param outgoingConnections
     * @return matrix outgoingConnections x (incomingConnections + 1 ), plus one comes from bias terms
     */
    public static SimpleMatrix randInitializeWeights(int incomingConnections, int outgoingConnections) {
        double epsilonInit = 0.12;
        Random rand = new Random();
        return SimpleMatrix.random(outgoingConnections, incomingConnections + 1,
                -epsilonInit, epsilonInit, rand);
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
    public static CostGradientTuple nnCostFunction(SimpleMatrix nn_params, int input_layer_size, int hidden_layer_size, int num_labels,
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

    /**
     * Predict labels for given set of points and weights.
     * @param theta1 neural network nets between input and hidden layer
     * @param theta2 neural network nets between hidden and output layer
     * @param x set of data points
     * @return predicted labels for input data points
     */
    public static SimpleMatrix predict(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix x) {
        // Setup some useful variables
        int m = x.numRows();
        int num_labels = theta2.numRows();

        SimpleMatrix ones = new SimpleMatrix(m, 1);
        ones.set(1.0);
        SimpleMatrix X = ones.combine(0,1,x);
        SimpleMatrix h1 = sigmoid(X.mult(theta1.transpose()));
        SimpleMatrix h2 = sigmoid(ones.combine(0, 1, h1).mult(theta2.transpose()));


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

    /**
     * Compares element-wise two matrices. c_{ij} = (Math.abs(a_{ij} - b_{ij}) < eps) ? 1.0 : 0.0
     * @param a
     * @param b
     * @return
     */
    public static SimpleMatrix elementEquals(SimpleMatrix a, SimpleMatrix b) {
        SimpleMatrix c = a.minus(b);

        for (int i =0;i<c.numRows();i++) {
            for (int j =0;j<c.numCols();j++) {
                double val = c.get(i,j);

                if (Math.abs(val) < 0.00001) {
                    c.set(i,j, 1.0);
                }
                else {
                    c.set(i,j, 0.0);
                }
            }
        }

        return c;
    }

    /**
     * Load matrix from file.
     * @param filename
     * @param lineSeparator character separating rows in matrix
     * @param itemSeparator character separating columns in matrix
     * @return matrix
     * @throws Exception
     */
    public static SimpleMatrix loadFile(String filename, String lineSeparator, String itemSeparator) throws Exception {
        byte[] encoded = Files.readAllBytes(Paths.get(filename));
        return Utils.parseString(new String(encoded, UTF_8), lineSeparator, itemSeparator);
    }

    /**
     * Converts raw text to matrix.
     * @param text text containg matrix
     * @param lineSeparator character separating rows in matrix
     * @param itemSeparator character separating columns in matrix
     * @return matrix
     * @throws Exception
     */
    private static SimpleMatrix parseString(String text, String lineSeparator, String itemSeparator) throws Exception {
        StringTokenizer tokRow;
        StringTokenizer tokCol;
        int rows;
        int cols;

        // determine dimenions
        tokRow = new StringTokenizer(text, lineSeparator);
        rows = tokRow.countTokens();
        tokCol = new StringTokenizer(tokRow.nextToken(), itemSeparator);
        cols = tokCol.countTokens();


        SimpleMatrix matrix = new SimpleMatrix(rows,cols);
        tokRow = new StringTokenizer(text, lineSeparator);
        rows = 0;
        while (tokRow.hasMoreTokens()) {
            tokCol = new StringTokenizer(tokRow.nextToken(), itemSeparator);
            cols = 0;
            while (tokCol.hasMoreTokens()) {
                matrix.set(rows, cols, Double.parseDouble(tokCol.nextToken()));
                cols++;
            }
            rows++;
        }

        return matrix;
    }

    /**
     * Load matrix from file.
     * @param filename
     * @return matrix
     * @throws Exception
     */
    public static SimpleMatrix loadFile(String filename) throws Exception {
        return Utils.loadFile(filename,  "\n", " ");
    }

    public static SimpleMatrix trainNetwork(SimpleMatrix X, SimpleMatrix y, int hidden_layer_size, int num_labels, double lambda, int iterations) {
        int input_layer_size = X.numCols(); // number of features per point
        CostFunction costFunction = new CostFunction(X,y,lambda,input_layer_size,hidden_layer_size,num_labels);

        SimpleMatrix theta1 = Utils.randInitializeWeights(input_layer_size,hidden_layer_size);
        // theta2 is the set of weights corresponding to connections from hidden layer to output layer
        SimpleMatrix theta2 = Utils.randInitializeWeights(hidden_layer_size, num_labels);
        // unrolling them into a linearized matrix
        SimpleMatrix nn_params = Utils.unroll(theta1).combine(0, theta1.getNumElements(), Utils.unroll(theta2));

        // optimizing set of weights
        SimpleMatrix optimized_nn_params = Fmincg.minimizeFunction(costFunction,nn_params,iterations);

        return optimized_nn_params;
    }

    public static void downloadHumanActivityRecognitionData() throws Exception {
        URL website = new URL("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip");
        ReadableByteChannel rbc = Channels.newChannel(website.openStream());
        FileOutputStream fos = new FileOutputStream("data.zip");
        fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        fos.close();

        ZipFile zipFile = new ZipFile("data.zip");
        zipFile.extractAll(".");
    }

}
