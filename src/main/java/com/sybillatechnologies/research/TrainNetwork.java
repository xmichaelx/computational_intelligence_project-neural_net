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
            SimpleMatrix testX = Utils.loadFile(path + "test/X_test.txt"),testy = Utils.loadFile(path + "test/y_test.txt") ;

            double lambda =  0; // parameters
            int iterations = 100,input_layer_size = X.numCols(),  hidden_layer_size = 25, num_labels = 6;

            SimpleMatrix optimized_nn_params  = Utils.trainNetwork(X,y,hidden_layer_size, num_labels, lambda,iterations);
            // recreate optimized layer weights
            SimpleMatrix theta1 = optimized_nn_params.extractMatrix(0,1,0,(input_layer_size+1)*hidden_layer_size);
            theta1.reshape(hidden_layer_size, input_layer_size + 1);

            SimpleMatrix theta2 = optimized_nn_params.extractMatrix(0,1,(input_layer_size+1)*hidden_layer_size,optimized_nn_params.getNumElements());
            theta2.reshape(num_labels, hidden_layer_size + 1);

            // prediction step - using learned weight to predict outputs for each point in test set
            SimpleMatrix prediction = Utils.predict(theta1, theta2, testX);

            // everywhere when we have a match
            SimpleMatrix matches = Utils.elementEquals(testy,prediction);
            System.out.println("Correctly classified: " + matches.elementSum() / matches.getNumElements());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
