
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
    mwIndex *jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
    mwIndex *ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt

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

    // The outer loop
    for (k = 0; k < iters; k++)
    {
        // Evaluate function value if output requested
        if (evalf == true) {
            hist[k] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
        }

        for (i = 0; i < d; i++) { wold[i] = w[i]; ym[i] = w[i]; zm[i] = w[i]; ww[i] = 0; last_seen[i] = 0;}

        // Initially, compute full gradient at current point w
        compute_full_gradient_sparse(Xt, wold, y, gold, n, d, lambda, ir, jc);

        double bb = 0;
        // The inner loop
        for (i = 0; i < m[k]; i++) {
            idx = *(iVals++); // Sample function and move pointer

            // lazy_update_Katyusha(w, wold, ww, gold, last_seen, stepSize, lambda, i, ir, jc + idx, ym, zm, tau, tau1, tau2);
            for (j = jc[idx]; j < jc[idx+1]; j++) {
                zm_prev[ir[j]] = zm[ir[j]];
                zm[ir[j]] -= stepSize * (i - last_seen[ir[j]]) * (gold[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));
                ym[ir[j]] = w[ir[j]] + (zm[ir[j]] - zm_prev[ir[j]]) * tau;
                w[ir[j]] = zm[ir[j]] * tau + wold[ir[j]] * 0.5 + ym[ir[j]] * tau1;
                // ww[ir[j]] += (i - last_seen[ir[j]]) * ym[ir[j]] * pow(tau2, i);
                ww[ir[j]] += (i - last_seen[ir[j]]) * ym[ir[j]];
                last_seen[ir[j]] = i;
            }

            // Compute current and old scalar sigmoid of the same example
            sigmoid = compute_sigmoid_sparse(Xt + jc[idx], w, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);
            sigmoidold = compute_sigmoid_sparse(Xt + jc[idx], wold, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);
            // Update the test point
            // update_test_point_sparse_Katyusha(Xt + jc[idx], w, ww, sigmoid,
            //     sigmoidold, jc[idx + 1] - jc[idx], stepSize, ym, zm, zm_prev, tau, tau2, ir + jc[idx], jc, i);
            for (j = jc[idx]; j < jc[idx+1]; j++) {
                zm[ir[j]] -= stepSize * (Xt[j] * (sigmoid - sigmoidold));
            }

            bb += pow(tau2, i);
            // for (long j = 0; j < d; j ++){ w_accu[j] += w[j]; }
        }

        // Update the rest of lazy_updates
        // finish_lazy_updates_Katyusha(w, wold, ww, gold, last_seen, stepSize, lambda, m[k], d, ym, zm, tau, tau1, tau2);

        for (j = 0; j < d; j++) {
            zm_prev[j] = zm[j];
            zm[j] -= stepSize * (i - last_seen[j]) * (gold[j] + lambda * (w[j] - wold[j]));
            ym[j] = w[j] + (zm[j] - zm_prev[j]) * tau;
            // ww[j] += (i - last_seen[j]) * ym[j] * pow(tau2, m[k]);
            ww[j] += (i - last_seen[j]) * ym[j];
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
