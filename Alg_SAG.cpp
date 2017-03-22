
#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions_sparse.h"
#include "logistic_functions_dense.h"
#include "utilities_dense.h"
#include "utilities_sparse.h"

/*
	USAGE:
	hist = SAG(w, Xt, y, lambda, stepsize, iVals);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	stepSize - a step-size
	iVals(iters x 1) - sequence of examples to choose, between 0 and (n-1)
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
	Computed ONLY if asked for output in MATALB.
*/

/// SAG_dense runs the S2GD algorithm for solving regularized 
/// logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SAG_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals;

	// Other variables
	long i, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	double sigmoid; // Scalar value of the derivative of sigmoid function
	double *sigmoidold; // Array of old sigmoids
	double *g; // Direction of update
	bool evalf = false; // set to true if function values should be evaluated

	double *hist; // Used to store function value at points in history

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[5]); // Number of outer iterations

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	sigmoidold = new double[n];
	g = new double[d];
	for (i = 0; i < n; i++) { sigmoidold[i] = 0; }
	for (i = 0; i < d; i++) { g[i] = 0; }
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (2 * n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SAG algorithm ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	for (k = 0; k < iters; k++)
	{
		// Evaluate function value from time to time if output requested
		if (evalf == true && k % (2 * n) == 0) {
			hist[(long)floor((double)k / (2 * n))] = compute_function_value(w, Xt, y, n, d, lambda);
		}

		idx = *(iVals++); // Sample function and move pointer

		// Compute current scalar sigmoid value of sampled function
		sigmoid = compute_sigmoid(Xt + d*idx, w, y[idx], d);

		// Update the aggregate gradient
		for (i = 0; i < d; i++) {
			g[i] += Xt[d*idx + i] * (sigmoid - sigmoidold[idx]) / n;
		}
		// Save the last sigmoid value for this function
		sigmoidold[idx] = sigmoid;
		
		// Update the test point
		update_test_point_dense_SAG(w, g, d, stepSize, lambda);
	}

	if (evalf == true) {
		hist[(long)floor((double)iters / (2 * n))] = compute_function_value(w, Xt, y, n, d, lambda);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] sigmoidold;
	delete[] g;

	if (evalf == true) { return plhs; }
	else { return 0; }

}

/// SAG_sparse runs the SAG algorithm for solving regularized 
/// logistic regression on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SAG_sparse(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals;

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	double sigmoid; // Scalar value of the derivative of sigmoid function
	double *sigmoidold; // Array of old sigmoids
	double *g; // Direction of update
	bool evalf = false; // set to true if function values should be evaluated

	long *last_seen; // used to do lazy "when needed" updates
	double *hist;// Used to store function value at points in history

	mwIndex *ir, *jc; // Used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	stepSize = mxGetScalar(prhs[4]); // Step-size (constant)
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[5]); // Number of outer iterations
	jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	sigmoidold = new double[n];
	g = new double[d];
	last_seen = new long[d];
	for (i = 0; i < n; i++) { sigmoidold[i] = 0; }
	for (i = 0; i < d; i++) { g[i] = 0; }
	for (i = 0; i < d; i++) { last_seen[i] = 0; }
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (2 * n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SAG algorithm ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	for (k = 0; k < iters; k++)
	{
		// Evaluate function value from time to time if output requested
		if (evalf == true && k % (2 * n) == 0) {
			hist[(long)floor((double)k / (2 * n))] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
		}

		idx = *(iVals++); // Sample function and move pointer

		// Update what we didn't in last few iterations
		// Only relevant coordinates
		update_test_point_sparse_SAG(w, g, last_seen, jc[idx + 1] - jc[idx], stepSize, lambda, ir + jc[idx], k);

		// Compute current scalar sigmoid value of sampled function
		sigmoid = compute_sigmoid_sparse(Xt + jc[idx], w, y[idx], jc[idx + 1] - jc[idx], ir + jc[idx]);
		
		// Update the aggregate gradient
		for (j = jc[idx]; j < jc[idx + 1]; j++) {
			g[ir[j]] += Xt[j] * (sigmoid - sigmoidold[idx]) / n;
		}
		// Save the last sigmoid value for this function
		sigmoidold[idx] = sigmoid;

		// Do NOT update test point. Relevant coordinates will be 
		// updated at the beginning of iterations when they will be needed.
	}

	// Finish lazy updates
	for (i = 0; i < d; i++) {
		w[i] -= stepSize * (iters - last_seen[i]) * (g[i] + lambda * w[i]);
	}

	// Evaluate the final function value
	if (evalf == true) {
		hist[(long)floor((double)iters / (2 * n))] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] sigmoidold;
	delete[] g;
	delete[] last_seen;

	//////////////////////////////////////////////////////////////////
	/// Return value /////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	if (evalf == true) { return plhs; }
	else { return 0; }

}

/// Entry function of MATLAB
/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
/// For more info about this syntax see 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/gateway-routine.html
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// First determine, whether the data matrix is stored in sparse format.
	// If it is, use more efficient algorithm
	if (mxIsSparse(prhs[1])) {
		plhs[0] = SAG_sparse(nlhs, prhs);
	}
	else {
		plhs[0] = SAG_dense(nlhs, prhs);
	}
}