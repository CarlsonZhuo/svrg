///  Copyright [2014] [Jakub Konecny]
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
/// 
///     http://www.apache.org/licenses/LICENSE-2.0
/// 
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.

//////////////////////////////////////////////////////////////////////
/// Jakub Konecny | www.jakubkonecny.com /////////////////////////////
/// last update : 8 July 2014            /////////////////////////////
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//// This header contains helper functions for algorithms     ////
//// included in experiments for the S2GD paper               ////
//// (Semi-Stochastic Gradient Descent Methods)               ////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

/// compute_sigmoid_sparse computes the derivative of logistic loss,
/// i.e. ~ exp(x) / (1 + exp(x)) for sparse data --- sparse x
/// *x - pointer to the first element of the data point
///      (e.g. Xt+jc[i] for i-th example)
/// *w - test point
/// y - label of the training example
/// d - number of nonzeros for current example (*x)
/// *ir - contains row indexes of elements of *x
///		  pointer to the first element of the array 
///		  (e.g. ir+jc[i] for i-th example)
double compute_sigmoid_sparse(double *x, double *w, double y,
	long d, mwIndex *ir)
{
	double tmp = 0;
	// Sparse inner product
	for (long j = 0; j < d; j++) {
		tmp += w[ir[j]] * x[j];
	}
	tmp = exp(y * tmp);
	tmp = y * tmp / (1 + tmp);
	return tmp;
}

/// compute_full_gradient computes the gradient of the entire function,
/// for sparse data matrix. Gradient is changed in place in g. 
/// *Xt - sparse data matrix; examples in columns!
/// *w - test point
/// *y - set of labels
/// *g - gradient; updated in place; input value irrelevant
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
/// For more info about ir, jc convention, see "Sparse Matrices" in 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/matlab-data.html
void compute_full_gradient_sparse(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_sigmoid_sparse(Xt + jc[i], w, y[i], jc[i + 1] - jc[i], ir + jc[i]);
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			g[ir[j]] += Xt[j] * sigmoid;
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
		g[i] += lambda * w[i];
	}
}