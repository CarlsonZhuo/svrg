
#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions_sparse.h"
#include "logistic_functions_dense.h"
#include "utilities_dense.h"
#include "utilities_sparse.h"

/*
	USAGE:
	hist = SGD(w, Xt, y, lambda, stepsize, iVals);
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

/// SGD_dense runs the SGD algorithm with constant stepsize for 
/// solving regularized logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SGD_dense(int nlhs, const mxArray *prhs[])
{

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals;
	
	// Other variables
	long k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters;
	long long idx;
	double sigmoid;

	double *hist; // Used to store function value at points in history
	bool evalf = false; // set to true if function values should be evaluated

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

	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SGD algorithm ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	for (k = 0; k < iters; k++) {
		// Evaluate function value from time to time if output requested
		if (evalf == true && k % (n) == 0) {
			hist[(long)floor((double)k / (n))] = compute_function_value(w, Xt, y, n, d, lambda);
		}

		idx = *(iVals++); // Sample function and move pointer

		sigmoid = compute_sigmoid(Xt + d*idx, w, y[idx], d);
		update_test_point_dense_SGD(Xt + d*idx, w, d, 
									sigmoid, stepSize, lambda);
	}

	// Evaluate the final function value
	if (evalf == true) {
		hist[(long)floor((double)iters / (n))] = compute_function_value(w, Xt, y, n, d, lambda);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	/// Return value /////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	if (evalf == true) { return plhs; }
	else { return 0; }

}

/// SGD_sparse runs the SGD algorithm with constant stepsize for 
/// solving regularized logistic regression on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SGD_sparse(int nlhs, const mxArray *prhs[])
{

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals;

	// Other variables
	long k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	double sigmoid; // Scalar value of the derivative of sigmoid function
	double *hist; // Used to store function value at points in history
	bool evalf = false; // set to true if function values should be evaluated

	long *last_seen; // used to do lazy "when needed" updates

	mwIndex *ir, *jc; // used to access nonzero elements of Xt
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

	last_seen = new long[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SGD algorithm ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	for (long j = 0; j < d; j++) { last_seen[j] = 0; }

	for (k = 0; k < iters; k++) {
		// Evaluate function value from time to time if output requested
		if (evalf == true && k % (n) == 0) {
			hist[(long)floor((double)k / (n))] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
		}
		
		idx = *(iVals++); // Sample function and move pointer

		lazy_update_SGD(w, last_seen, stepSize, lambda, k, ir, jc + idx);

		sigmoid = compute_sigmoid_sparse(Xt + jc[idx], w, y[idx], 
							jc[idx + 1] - jc[idx], ir + jc[idx]);
		update_test_point_sparse_SGD(Xt + jc[idx], w, jc[idx + 1] - jc[idx],
							sigmoid, stepSize, lambda, ir + jc[idx]);
	}

	// Evaluate the final function value
	if (evalf == true) {
		hist[(long)floor((double)iters / (n))] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

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
		plhs[0] = SGD_sparse(nlhs, prhs);
	}
	else {
		plhs[0] = SGD_dense(nlhs, prhs);
	}


}