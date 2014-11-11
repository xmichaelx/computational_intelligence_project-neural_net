package com.sybillatechnologies.research;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Michał Drzał on 2014-11-09.
 * Based on Matlab Code from exercise 4 from Machine Learning Course by Andrew Ng on Coursera
 */

public class TrainNetwork {
    public static void main(String[] args) {
        try {
            // load matrices from files
            String path = "UCI HAR Dataset/";
            SimpleMatrix X = Utils.loadFile(path + "train/X_train.txt"), y = Utils.loadFile(path + "train/y_train.txt");

            // parameters
            double lambda =  0;
            int iterations = 100,input_layer_size = X.numCols(),  hidden_layer_size = 25, num_labels = 6;

            NeuralNetwork network = new NeuralNetwork(input_layer_size, hidden_layer_size, num_labels);
            // training step
            network.train(X,y,lambda,iterations);
            // prediction step - using learned weight to predict outputs for each point in test set

            SimpleMatrix testX = Utils.loadFile(path + "test/X_test.txt"),testy = Utils.loadFile(path + "test/y_test.txt") ;
            SimpleMatrix prediction = network.predict(testX);

            // everywhere when we have a match
            SimpleMatrix matches = Utils.elementEquals(testy,prediction);
            System.out.println("Correctly classified: " + matches.elementSum() / matches.getNumElements());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
