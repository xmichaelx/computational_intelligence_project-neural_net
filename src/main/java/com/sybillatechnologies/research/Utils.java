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
