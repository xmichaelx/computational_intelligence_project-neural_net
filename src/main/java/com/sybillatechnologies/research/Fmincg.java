package com.sybillatechnologies.research;

import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by Michał Drzał on 2014-11-09.
 * Copied from https://github.com/thomasjungblut/thomasjungblut-common/blob/master/src/de/jungblut/math/minimize
 * which in turn was based on Matlab Code from one of assignments from Machine Learning Course by Andrew Ng.
 */
public class Fmincg {
    // extrapolate maximum 3 times the current bracket.
    // this can be set higher for bigger extrapolations
    public static double EXT = 3.0;
    // a bunch of constants for line searches
    private static final double RHO = 0.01;
    // RHO and SIG are the constants in the Wolfe-Powell conditions
    private static final double SIG = 0.5;
    // don't reevaluate within 0.1 of the limit of the current bracket
    private static final double INT = 0.1;
    // max 20 function evaluations per line search
    private static final int MAX = 20;
    // maximum allowed slope ratio
    private static final int RATIO = 100;

    /**
     * Minimizes the given CostFunction with Nonlinear conjugate gradient method. <br/>
     * It uses the Polack-Ribiere (PR) to calculate the conjugate direction. See <br/>
     * http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method <br/>
     * for more information.
     *
     * @param f the cost function to minimize.
     * @param theta input vector, unrolled layers of neural net
     * @param maxIterations the number of iterations to make
     * @return a tuple containing the optimized input
     */
    public static SimpleMatrix minimizeFunction(CostFunction f, SimpleMatrix theta, int maxIterations) {
        return new Fmincg().minimize(f, theta, maxIterations);
    }

    private SimpleMatrix minimize(CostFunction f, SimpleMatrix theta, int length) {
        SimpleMatrix input = theta;

        int M = 0;
        int i = 0; // zero the run length counter
        int red = 1; // starting point
        int ls_failed = 0; // no previous line search has failed
        CostGradientTuple evaluateCost = f.getCost(input);
        double f1 = evaluateCost.getJ();
        SimpleMatrix df1 = evaluateCost.getGrad();
        i += (length < 0 ? 1 : 0);

        SimpleMatrix s = df1.negative();

        double d1 = s.negative().dot(s);
        double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

        while (i < Math.abs(length)) {
            i = i + (length > 0 ? 1 : 0);// count iterations?!

            SimpleMatrix X0 = input.copy();
            double f0 = f1;
            SimpleMatrix df0 = df1.copy();
            CommonOps.addEquals(input.getMatrix(), z1,s.getMatrix());
            CostGradientTuple evaluateCost2 = f.getCost(input);
            double f2 = evaluateCost2.getJ();
            SimpleMatrix df2 = evaluateCost2.getGrad();

            i = i + (length < 0 ? 1 : 0); // count epochs
            double d2 = df2.dot(s);

            double f3 = f1;
            double d3 = d1;
            double z3 = -z1;

            if (length > 0) {
                M = MAX;
            } else {
                M = Math.min(MAX, -length - i);
            }

            int success = 0;
            double limit = -1;

            while(true) {
                while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {
                    // tighten the bracket
                    limit = z1;
                    double z2 = 0.0d;
                    double A = 0.0d;
                    double B = 0.0d;

                    if (f2 > f1) {
                        // quadratic fit
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                    } else {
                        // cubic fit
                        A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                        B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                        // numerical error possible - ok!
                        z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                    }

                    if (Double.isNaN(z2) || Double.isInfinite(z2)) {
                        // if we had a numerical problem then bisect
                        z2 = z3 / 2.0d;
                    }

                    // don't accept too close to limits
                    z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
                    // update the step
                    z1 = z1 + z2;
                    CommonOps.addEquals(input.getMatrix(), z2,s.getMatrix());

                    CostGradientTuple evaluateCost3 = f.getCost(input);
                    f2 = evaluateCost3.getJ();
                    df2 = evaluateCost3.getGrad();

                    M = M - 1;
                    i = i + (length < 0 ? 1 : 0); // count epochs
                    d2 = df2.dot(s);
                    // z3 is now relative to the location of z2
                    z3 = z3 - z2;
                }

                if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
                    break; // this is a failure
                } else if (d2 > SIG * d1) {
                    success = 1;
                    break; // success
                } else if (M == 0) {
                    break; // failure
                }

                // make cubic extrapolation
                double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));
                // num prob or wrong sign?
                if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0) {
                    // if we have no upper limit
                    if (limit < -0.5) {
                        // the extrapolate the maximum amount
                        z2 = z1 * (EXT - 1);
                    } else {
                        // otherwise bisect
                        z2 = (limit - z1) / 2;
                    }
                } else if ((limit > -0.5) && (z2 + z1 > limit)) {
                    // extraplation beyond max?
                    z2 = (limit - z1) / 2; // bisect
                } else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
                    // extrapolationbeyond limit
                    z2 = z1 * (EXT - 1.0); // set to extrapolation limit
                } else if (z2 < -z3 * INT) {
                    z2 = -z3 * INT;
                } else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
                    // too close to the limit
                    z2 = (limit - z1) * (1.0 - INT);
                }

                // set point 3 equal to point 2
                f3 = f2;
                d3 = d2;
                z3 = -z2;
                z1 = z1 + z2;
                // update current estimates
                CommonOps.addEquals(input.getMatrix(), z2,s.getMatrix());
                final CostGradientTuple evaluateCost3 = f.getCost(input);
                f2 = evaluateCost3.getJ();
                df2 = evaluateCost3.getGrad();
                M = M - 1;
                i = i + (length < 0 ? 1 : 0); // count epochs?!
                d2 = df2.dot(s);
            }// end of line search

            SimpleMatrix tmp = null;
            if (success == 1) { // if line search succeeded
                f1 = f2;

                System.out.println("Iteration " + i + " | Cost: " + f1);
                /*if (verbose) {
                    LOG.info("Iteration " + i + " | Cost: " + f1);
                    onIterationFinished(i, f1, input);
                }*/

                // Polack-Ribiere direction: s =
                // (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
                double numerator = (df2.dot(df2) - df1.dot(df2)) / df1.dot(df1);

                s = s.scale(numerator).minus(df2);
                tmp = df1;
                df1 = df2;
                df2 = tmp; // swap derivatives
                d2 = df1.dot(s);
                if (d2 > 0) { // new slope must be negative
                    s = df1.scale(-1.0d); // otherwise use steepest direction
                    d2 = s.scale(-1.0d).dot(s);
                }
                // realmin in octave = 2.2251e-308
                // slope ratio but max RATIO
                z1 = z1 * Math.min(RATIO, d1 / (d2 - 2.2251e-308));
                d1 = d2;
                ls_failed = 0; // this line search did not fail
            } else {
                input = X0;
                f1 = f0;
                df1 = df0; // restore point from before failed line search
                // line search failed twice in a row?
                if (ls_failed == 1 || i > Math.abs(length)) {
                    break; // or we ran out of time, so we give up
                }
                tmp = df1;
                df1 = df2;
                df2 = tmp; // swap derivatives
                s = df1.scale(-1.0d); // try steepest
                d1 = s.scale(-1.0d).dot(s);
                z1 = 1.0d / (1.0d - d1);
                ls_failed = 1; // this line search failed
            }


        }

        return input;
    }

}
