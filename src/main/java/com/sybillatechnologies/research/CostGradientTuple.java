package com.sybillatechnologies.research;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Michał Drzał on 2014-11-09.
 * based on Matlab Code from one of assignments from Machine Learning Course by Andrew Ng
 */
public class CostGradientTuple {
    private SimpleMatrix grad;
    private double J;

    public SimpleMatrix getGrad() {
        return grad;
    }

    public double getJ() {
        return J;
    }

    public CostGradientTuple(SimpleMatrix grad, double j) {
        this.grad = grad;
        J = j;
    }
}
