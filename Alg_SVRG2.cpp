
#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions_sparse.h"
#include "logistic_functions_dense.h"
#include "utilities_dense.h"
#include "utilities_sparse.h"

/*
    USAGE:
    hist = SVRG(w, Xt, y, lambda, stepsize, iVals, m);
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
mxArray* SVRG_dense(int nlhs, const mxArray *prhs[]) {

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

    double *wold    = mxGetPr(prhs[0]); // The variable to be learned
    double *Xt      = mxGetPr(prhs[1]); // Data matrix (transposed)
    double *y       = mxGetPr(prhs[2]); // Labels
    double lambda   = mxGetScalar(prhs[3]); // Regularization parameter
    double stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
    long long *iVals= (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
    long long *m    = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
    
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
    double *w_accu = new double[d];
    double *w = new double[d]; for(k=0;k<d;k++){w[k]=0;}
    double *gold = new double[d];
    if (evalf == true) {
        plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }

    //////////////////////////////////////////////////////////////////
    /// The SVRG algorithm ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // The outer loop
    for (k = 0; k < iters; k++)
    {
        // Evaluate function value if output requested
        if (evalf == true) {
            hist[k] = compute_function_value(wold, Xt, y, n, d, lambda);
        }

        // for (i = 0; i < d; i++) { w[i] = wold[i]; }

        // Initially, compute full gradient at current point w
        compute_full_gradient(Xt, wold, y, gold, n, d, lambda);

        for (i = 0; i < d; i++) { w_accu[i] = 0; }

        // The inner loop
        for (i = 0; i < m[k]; i++) {
            idx = *(iVals++); // Sample function and move pointer

            // Compute current and old scalar sigmoid of the same example
            sigmoid = compute_sigmoid(Xt + d*idx, w, y[idx], d);
            sigmoidold = compute_sigmoid(Xt + d*idx, wold, y[idx], d);

            // Update the test point
            update_test_point_dense_SVRG(Xt + d*idx, w, wold, gold, 
                sigmoid, sigmoidold, d, stepSize, lambda);

            for (long j = 0; j < d; j ++){ w_accu[j] += w[j]; }
        }

        // for (i = 0; i < d; i++) { wold[i] = w[i]; }
        for (i = 0; i < d; i++) { wold[i] = w_accu[i]/m[k]; }
    }

    if (evalf == true) {
        hist[iters] = compute_function_value(w, Xt, y, n, d, lambda);
    }


    delete[] w_accu;
    delete[] w;
    delete[] gold;

    if (evalf == true) { return plhs; }
    else { return 0; }
    
}

/// nlhs - number of output parameters requested
///        if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SVRG_sparse(int nlhs, const mxArray *prhs[]) {

    //////////////////////////////////////////////////////////////////
    /// Declare variables ////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // Other variables
    long i, j, k; // Some loop indexes
    long long idx; // For choosing indexes
    // Scalar value of the derivative of sigmoid function
    double sigmoid, sigmoidold;
    bool evalf = false; // set to true if function values should be evaluated

    double *hist; // Used to store function value at points in history

    mxArray *plhs; // History array to return if needed

    double *w       = mxGetPr(prhs[0]); // The variable to be learned
    double *Xt      = mxGetPr(prhs[1]); // Data matrix (transposed)
    double *y       = mxGetPr(prhs[2]); // Labels
    double lambda   = mxGetScalar(prhs[3]); // Regularization parameter
    double stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
    long long *iVals= (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
    long long *m    = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
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
    double *wold = new double[d];
    double *gold = new double[d];
    long *last_seen = new long[d];
    if (evalf == true) {
        plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }

    //////////////////////////////////////////////////////////////////
    /// The SVRG algorithm ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////

    // The outer loop
    for (k = 0; k < iters; k++)
    {
        // Evaluate function value if output requested
        if (evalf == true) {
            hist[k] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
        }

        // Initially, compute full gradient at current point w.
        compute_full_gradient_sparse(Xt, w, y, gold, n, d, lambda, ir, jc);
        // Save the point where full gradient was computed; initialize last_seen
        for (j = 0; j < d; j++) { wold[j] = w[j]; last_seen[j] = 0; }

        // The inner loop
        for (i = 0; i < m[k]; i++) {
            idx = *(iVals++); // Sample function and move pointer

            // Update what we didn't in last few iterations
            // Only relevant coordinates
            lazy_update_SVRG(w, wold, gold, last_seen, stepSize, lambda, i, ir, jc + idx);

            // Compute current and old scalar sigmoid of the same example
            sigmoid = compute_sigmoid_sparse(Xt + jc[idx], w, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);
            sigmoidold = compute_sigmoid_sparse(Xt + jc[idx], wold, y[idx], 
                                jc[idx + 1] - jc[idx], ir + jc[idx]);

            // Update the test point
            update_test_point_sparse_SVRG(Xt + jc[idx], w, sigmoid,
                sigmoidold, jc[idx + 1] - jc[idx], stepSize, ir + jc[idx]);
        }

        // Update the rest of lazy_updates
        finish_lazy_updates_SVRG(w, wold, gold, last_seen, stepSize, lambda, m[k], d);
    }

    // Evaluate the final function value
    if (evalf == true) {
        hist[iters] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
    }


    delete[] wold;
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
        plhs[0] = SVRG_sparse(nlhs, prhs);
    } else {
        plhs[0] = SVRG_dense(nlhs, prhs);
    }
}
