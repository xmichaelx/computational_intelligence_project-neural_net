package com.sybillatechnologies.research;

/**
 * Created by Dawid on 2014-11-26.
 */
public class PolynomialChainFitting
{
    private static final int CHAIN_LENGTH = 8;
    private static final int NUMBER_OF_KNOTS = 4;
    private static final int ORDER_OF_POLYNOMIAL = 2; // we're using only quadratic functions

    private double[] findKnots(double[][] data) throws KnotsSearchingException
    {
        int i, j;
        double average = 0.0;
        int chains = 0;
        int chain[][] = new int[2][3]; // len,start,end

        for(i = 0; i < data.length; ++i)
        {
            average += data[i][1];
        }

        average /= (double) data.length;

        i = 0;
        while(data[i][1] < average)
        {
            ++i;
        }

        for(; i < data.length; ++i)
        {
            if(data[i][1] > average)
            {
                continue;
            }

            boolean isChainTooShort = false;
            boolean isChainWrapped = false;

            for(j = 1; j < CHAIN_LENGTH; ++j)
            {
                if(i + j == data.length)
                {
                    i = -j;
                    isChainWrapped = true;
                }

                if(data[i + j][1] > average)
                {
                    i += j;
                    isChainTooShort = true;
                    break;
                }
            }

            if(isChainWrapped && isChainTooShort)
            {
                break;
            }

            if(isChainTooShort)
            {
                continue;
            }

            while(data[i + j][1] < average)
            {
                if(i + j == data.length)
                {
                    i = -j;
                    isChainWrapped = true;
                }
                ++j;
            }

            if(chains < 2)
            {
                ++chains;
                chain[chains - 1][0] = j - 1;
                chain[chains - 1][1] = isChainWrapped ? data.length + i : i;
                chain[chains - 1][2] = i + j - 1;
            }
            else
            {
                ++chains;
                if(j - 1 > chain[0][0])
                {
                    if(chain[0][0] > chain[1][0])
                    {
                        chain[1][0] = chain[0][0];
                        chain[1][1] = chain[0][1];
                        chain[1][2] = chain[0][2];
                    }

                    chain[0][0] = j - 1;
                    chain[0][1] = isChainWrapped ? data.length + i : i;
                    chain[0][2] = i + j - 1;
                }
                else if(j - 1 > chain[1][0])
                {
                    chain[1][0] = j - 1;
                    chain[1][1] = isChainWrapped ? data.length + i : i;
                    chain[1][2] = i + j - 1;
                }
            }

            i += j - 1;
            if(isChainWrapped)
            {
                break;
            }
        }

        if(chains < 2)
        {
            throw new KnotsSearchingException("Number of chains less than 2");
        }

        double[] knots = new double[NUMBER_OF_KNOTS];

        return knots;
    }

    private double fitChain(double[][] data, double[] knots)
    {
        int i, j, k, intervals, kfinal, cum, int1index;
        double chisq, chi2tot, knot, dknot;

        int[] n = new int[NUMBER_OF_KNOTS];
        i = 0; cum = 0;
        while(data[i][0] < knots[0])
        {
            i++;
            cum++;
        }
        n[NUMBER_OF_KNOTS - 1] = i;
        int1index = cum;

        for(j = 0; j < NUMBER_OF_KNOTS - 1; j++)
        {
            while(data[i][0] < knots[j + 1] && i < data.length)
            {
                i++;
            }
            n[j] += i - cum;
            if(n[j] <= ORDER_OF_POLYNOMIAL)
            {
                return 1e10;
            }
            cum += n[j];
        }

        n[j] += data.length - cum;
        if(n[j] <= ORDER_OF_POLYNOMIAL)
        {
            return 1e10;
        }

        double[] x = new double[NUMBER_OF_KNOTS];
        double[] y = new double[NUMBER_OF_KNOTS];
        double[] w = new double[NUMBER_OF_KNOTS];

        cum = 0;
        for(j = 0; j < NUMBER_OF_KNOTS - 1; ++j)
        {
            for(i = 0; i < n[j]; i++)
            {
                x[j] = data[int1index + cum + i][0] - knots[j];
                y[j] = data[int1index + cum + i][1];
                w[j] = data[int1index + cum + i][2];
            }
            cum += n[j];
        }

        knot = knots[j];
        for(i = 0; i < n[j]; ++i)
        {
            if(int1index + cum + i == data.length)
            {
                cum = -int1index - i;
                knot -= 1.0;
            }

            x[j] = data[int1index + cum + i][0] - knot;
            y[j] = data[int1index + cum + i][1];
            w[j] = data[int1index + cum + i][2];
        }

        if(ORDER_OF_POLYNOMIAL == 1)
        {
            intervals = NUMBER_OF_KNOTS - 1;
        }
        else
        {
            intervals = NUMBER_OF_KNOTS;
        }

        // todo: the rest of fitChain method
        return -1;
    }
}
