
#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions_sparse.h"
#include "logistic_functions_dense.h"
#include "utilities_dense.h"
#include "utilities_sparse.h"

/*
    USAGE:
    hist = Katyusha(w, Xt, y, lambda, stepsize, iVals, m);
    ==================================================================
    INPUT PARAMETERS:
    w (d x 1) - initial point; updated in place
    Xt (d x n) - data matrix; transposed (data points are columns); real
    y (n x 1) - labels; in {-1,1}
    lambda - scalar regularization param
    stepSize - a step-size
    iVals (sum(m) x 1) - sequence of examples to choose, between 0 and (n-1)
    m (iters x 1) - sizes of the inner loops
    ==================================================================
    OUTPUT PARAMETERS:
    hist = array of function values after each outer loop.
           Computed ONLY if explicitely asked for output in MATALB.
*/


/// nlhs - number of output parameters requested
///        if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* Katyusha_dense(int nlhs, const mxArray *prhs[]) {

    //////////////////////////////////////////////////////////////////
    /// Declare variables ////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // Other variables
    long long i; // Some loop indexes
    long k; // Some loop indexes
    long long idx; // For choosing indexes
    double sigmoid, sigmoidold;
    bool evalf = false; // set to true if function values should be evaluated

    double *hist; // Used to store function value at points in history

    mxArray *plhs; // History array to return if needed

    double *w    = mxGetPr(prhs[0]); // The variable to be learned
    double *Xt      = mxGetPr(prhs[1]); // Data matrix (transposed)
    double *y       = mxGetPr(prhs[2]); // Labels
    double lambda   = mxGetScalar(prhs[3]); // Regularization parameter
    double stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
    long long *iVals= (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
    long long *m    = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
    double tau      = mxGetScalar(prhs[7]); // Tau
    double tau1     = 0.5 - tau;
    double tau2     = 1 + lambda*stepSize;
    
    if (nlhs == 1) {
        evalf = true;
    }

    if (!mxIsClass(prhs[5], "int64"))
        mexErrMsgTxt("iVals must be int64");
    if (!mxIsClass(prhs[6], "int64"))
        mexErrMsgTxt("m must be int64");

    long d = mxGetM(prhs[1]); // Number of features, or dimension of problem
    long n = mxGetN(prhs[1]); // Number of samples, or data points
    long iters = mxGetM(prhs[6]); // Number of outer iterations

    // Allocate memory to store full gradient and point in which it
    // was computed
    // double *w_accu = new double[d];
    double *wold = new double[d]; for(k=0;k<d;k++){wold[k]=0;}
    double *ww = new double[d]; for(k=0;k<d;k++){ww[k]=0;}
    double *ym = new double[d]; for(k=0;k<d;k++){ym[k]=0;}
    double *zm = new double[d]; for(k=0;k<d;k++){zm[k]=0;}
    double *zm_prev = new double[d]; for(k=0;k<d;k++){zm_prev[k]=0;}
    double *gold = new double[d];
    if (evalf == true) {
        plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }

    //////////////////////////////////////////////////////////////////
    /// The Katyusha algorithm ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // The outer loop
    for (k = 0; k < iters; k++)
    {
        // Evaluate function value if output requested
        if (evalf == true) {
            hist[k] = compute_function_value(w, Xt, y, n, d, lambda);
        }

        for (i = 0; i < d; i++) { wold[i] = w[i]; ym[i] = w[i]; zm[i] = w[i]; ww[i] = 0;}

        // Initially, compute full gradient at current point w
        compute_full_gradient(Xt, wold, y, gold, n, d, lambda);

        double bb = 0;
        // The inner loop
        for (i = 0; i < m[k]; i++) {
            idx = *(iVals++); // Sample function and move pointer

            for (long j = 0; j < d; j ++){
                w[j] = zm[j]*tau + wold[j]*0.5 + ym[j]*tau1;
            }

            // Compute current and old scalar sigmoid of the same example
            sigmoid = compute_sigmoid(Xt + d*idx, w, y[idx], d);
            sigmoidold = compute_sigmoid(Xt + d*idx, wold, y[idx], d);

            // Update the test point
            update_test_point_dense_Katyusha(Xt + d*idx, w, wold, gold, 
                sigmoid, sigmoidold, d, stepSize, lambda, ww, ym, zm, zm_prev, tau, tau2, i);

            bb += pow(tau2, i);
            // for (long j = 0; j < d; j ++){ w_accu[j] += w[j]; }
        }

        for (i = 0; i < d; i++) { w[i] = ww[i]/bb; }
        // for (i = 0; i < d; i++) { wold[i] = w_accu[i]/m[k]; }
    }

    if (evalf == true) {
        hist[iters] = compute_function_value(w, Xt, y, n, d, lambda);
    }


    // delete[] w_accu;
    delete[] wold;
    delete[] ww;
    delete[] ym;
    delete[] zm;
    delete[] zm_prev;
    delete[] gold;

    if (evalf == true) { return plhs; }
    else { return 0; }
    
}

/// nlhs - number of output parameters requested
///        if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments

mxArray* Katyusha_sparse(int nlhs, const mxArray *prhs[]) {

    //////////////////////////////////////////////////////////////////
    /// Declare variables ////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // Other variables
    long long i, j; // Some loop indexes
    long k; // Some loop indexes
    long long idx; // For choosing indexes
    double sigmoid, sigmoidold;
    bool evalf = false; // set to true if function values should be evaluated

    double *hist; // Used to store function value at points in history

    mxArray *plhs; // History array to return if needed

    double *w    = mxGetPr(prhs[0]); // The variable to be learned
    double *Xt      = mxGetPr(prhs[1]); // Data matrix (transposed)
    double *y       = mxGetPr(prhs[2]); // Labels
    double lambda   = mxGetScalar(prhs[3]); // Regularization parameter
    double stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
    long long *iVals= (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
    long long *m    = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
    double tau      = mxGetScalar(prhs[7]); // Tau
    double Lmax     = mxGetScalar(prhs[8]); // Lmax
    double tau1     = 0.5 - tau;
    double tau2     = 1 + lambda*stepSize;
    double tau3     = 0;

    if (nlhs == 1) {
        evalf = true;
    }

    if (!mxIsClass(prhs[5], "int64"))
        mexErrMsgTxt("iVals must be int64");
    if (!mxIsClass(prhs[6], "int64"))
        mexErrMsgTxt("m must be int64");


    long d = mxGetM(prhs[1]); // Number of features, or dimension of problem
    long n = mxGetN(prhs[1]); // Number of samples, or data points
    long iters = mxGetM(prhs[6]); // Number of outer iterations
    mwIndex *jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
    mwIndex *ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt

    Lmax    = 3*Lmax;
    tau     = fmin(0.5, sqrt(2*iters*lambda/Lmax));
    stepSize= 1 / (1.0*tau*Lmax); 
    tau1    = 1 + lambda*stepSize; 
    tau2    = Lmax + lambda;
    tau3    = Lmax/tau2;

    // Allocate memory to store full gradient and point in which it
    // was computed
    // double *w_accu = new double[d];
    double *wold = new double[d]; for(k=0;k<d;k++){wold[k]=0;}
    double *ww = new double[d]; for(k=0;k<d;k++){ww[k]=0;}
    double *ym = new double[d]; for(k=0;k<d;k++){ym[k]=0;}
    double *zm = new double[d]; for(k=0;k<d;k++){zm[k]=0;}
    double *zm_prev = new double[d]; for(k=0;k<d;k++){zm_prev[k]=0;}
    double *gold = new double[d];
    long *last_seen = new long[d];


    if (evalf == true) {
        plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }

    //////////////////////////////////////////////////////////////////
    /// The Katyusha algorithm ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    for (i = 0; i < d; i++) { ym[i] = w[i]; zm[i] = w[i];}

    // The outer loop
    for (k = 0; k < iters; k++)
    {
        // Evaluate function value if output requested
        if (evalf == true) {
            hist[k] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
        }

        // for (i = 0; i < d; i++) { wold[i] = w[i]; ym[i] = w[i]; zm[i] = w[i]; ww[i] = 0; last_seen[i] = 0;}
        for (i = 0; i < d; i++) { wold[i] = w[i]; ww[i] = 0; last_seen[i] = 0;}

        // Initially, compute full gradient at current point w
        compute_full_gradient_sparse(Xt, wold, y, gold, n, d, lambda, ir, jc);

        double bb = 0;
        // The inner loop
        for (i = 0; i < m[k]; i++) {
            idx = *(iVals++); // Sample function and move pointer

            // lazy_update_Katyusha(w, wold, ww, gold, last_seen, stepSize, lambda, i, ir, jc + idx, ym, zm, tau, tau1, tau2);
            for (j = jc[idx]; j < jc[idx+1]; j++) {
                // terms that did not consider during "lazy update"
                double tmp = (gold[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));

                // update zm
                zm[ir[j]] = (1/tau1) * zm[ir[j]] - (stepSize/tau1) * (tmp + zm_prev[ir[j]]);
                // the loop calculates the sum of a geometric progression
                // i.e. a(n) = (q^n) * a(0) - (1-q^(n-1))/(1-q) * C
                // where q = 1/tau1, C = stepSize/tau1 * tmp
                for (long ii = last_seen[ir[j]]+1; ii < i; ii++) {
                    zm[ir[j]] = (1/tau1) * zm[ir[j]] - (stepSize/tau1) * tmp;
                }

                // update ym
                // if ir[j] appeared in last iter
                // then (1/tau2) * zm_prev[ir[j]] is included in the subtrahend
                if (last_seen[ir[j]] == i-1) {
                    ym[ir[j]] = tau3 * w[ir[j]] - (1/tau2) * (zm_prev[ir[j]] + tmp);
                } else {
                    ym[ir[j]] = tau3 * w[ir[j]] - (1/tau2) * tmp;
                }

                // update ww
                ww[ir[j]] += (i - last_seen[ir[j]]) * (tau3 * w[ir[j]] - (1/tau2) * tmp);
                ww[ir[j]] -= (1/tau2) * zm_prev[ir[j]];

                // update w
                w[ir[j]] = zm[ir[j]] * tau + wold[ir[j]] * 0.5 + ym[ir[j]] * (0.5-tau);
                last_seen[ir[j]] = i;                
            }

            // Compute current and old scalar sigmoid of the same example
            sigmoid = compute_sigmoid_sparse(Xt + jc[idx], w, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);
            sigmoidold = compute_sigmoid_sparse(Xt + jc[idx], wold, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);

            // Update the test point
            for (j = jc[idx]; j < jc[idx+1]; j++) {
                zm_prev[ir[j]] = Xt[j] * (sigmoid - sigmoidold);
                // ww[ir[j]] += pow(tau1, i) * ym[ir[j]];
            }

            bb += pow(tau1, i);
            // for (long j = 0; j < d; j ++){ w_accu[j] += w[j]; }
        }

        // Update the rest of lazy_updates
        for (j = 0; j < d; j++) {
            double tmp = (gold[j] + lambda * (w[j] - wold[j]));

            // update zm
            zm[j] = 1/tau1 * zm_prev[j] - stepSize/tau1 * (tmp + zm_prev[j]);
            for (long ii = last_seen[j]+1; ii < i; ii++) {
                zm[j] = (1/tau1) * zm[j] - stepSize/tau1 * tmp;
            }

            // update ym
            if (last_seen[j] == i-1) {
                ym[j] = tau3 * w[j] - (1/tau2) * (zm_prev[j] + tmp);
            } else {
                ym[j] = tau3 * w[j] - (1/tau2) * tmp;
            }
            
            // update ww
            ww[j] += (i - last_seen[j]) * (tau3 * w[j] - (1/tau2) * tmp);
            ww[j] -= (1/tau2) * zm_prev[j];
        }

        for (i = 0; i < d; i++) { 
            // w[i] = ww[i]/bb; 
            w[i] = ww[i]/m[k]; 
        }
        // for (i = 0; i < d; i++) { wold[i] = w_accu[i]/m[k]; }
    }

    if (evalf == true) {
        hist[iters] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
    }


    // delete[] w_accu;
    delete[] wold;
    delete[] ww;
    delete[] ym;
    delete[] zm;
    delete[] zm_prev;
    delete[] gold;
    delete[] last_seen;


    if (evalf == true) { return plhs; }
    else { return 0; }

}

/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
/// For more info about this syntax see 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/gateway-routine.html
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (mxIsSparse(prhs[1])) {
        plhs[0] = Katyusha_sparse(nlhs, prhs);
    } else {
        plhs[0] = Katyusha_dense(nlhs, prhs);
    }
}
