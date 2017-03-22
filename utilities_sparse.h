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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//// This header contains helper functions for algorithms         ////
//// included in experiments for the S2GD paper				      ////
//// (Semi-Stochastic Gradient Descent Methods)                   ////
//// This contains functions for sparse data matrix               ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/// Compute the function value of average regularized logistic loss
/// *w - tesp point
/// *Xt - data matrix
/// *y - set of labels
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
double compute_function_value_sparse(double* w, double *Xt, double *y,
	long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			tmp += Xt[j] * w[ir[j]];
		}
		value += log(1 + exp(y[i] * tmp));
	}
	value = value / n;

	// Add regularization term
	for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j];
	}
	return value;
}

/// Updates the test point *w in place
/// Makes the step only in the nonzero coordinates of *x,
/// and without regularizer. The regularizer step is constant
/// across more iterations --- updated in lazy_updates
/// *x - training example
/// *w - test point; updated in place
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// d - number of nonzeros of training example *x
/// stepSize - stepsize parameter
/// *ir - row indexes of nonzero elements of *x
void update_test_point_sparse_SVRG(double *x, double *w,
	double sigmoid, double sigmoidold,
	long d, double stepSize, mwIndex *ir)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= stepSize * (x[j] * (sigmoid - sigmoidold));
	}
}

/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For S2GD algorithm
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// i - number of iteration from which this lazy update was called
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// *jc - index of element in data matrix where starts the training
///		  exampls for which the gradient is to be computed
void lazy_update_SVRG(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) *
			(g[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));
		last_seen[ir[j]] = i;
	}
}

/// Finises the "lazy" updates at the end of outer loop
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// iters - number of steps taken in the current outer loop
///			also size of the just finished inner loop
/// d - dimension of the problem
void finish_lazy_updates_SVRG(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long iters, long d)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (iters - last_seen[j]) *
			(g[j] + lambda * (w[j] - wold[j]));
	}
}

/// Updates the test point *w in place
/// Makes the step only in the nonzero coordinates of *x,
/// and without regularizer. The regularizer step is constant
/// across more iterations --- updated in lazy_updates
/// *x - training example
/// *w - test point; updated in place
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// d - number of nonzeros of training example *x
/// stepSize - stepsize parameter
/// *ir - row indexes of nonzero elements of *x
void update_test_point_sparse_ASVRG(double *x, double *w,
	double sigmoid, double sigmoidold,
	long d, double stepSize, mwIndex *ir, double *y, double tau)
{
	for (long j = 0; j < d; j++) {
		y[ir[j]] -= stepSize * (x[j] * (sigmoid - sigmoidold));
	}
}

/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For S2GD algorithm
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// i - number of iteration from which this lazy update was called
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// *jc - index of element in data matrix where starts the training
///		  exampls for which the gradient is to be computed
void lazy_update_ASVRG(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc,
	double *y, double tau)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		y[ir[j]] -= stepSize * (i - last_seen[ir[j]]) *
			(g[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));
		w[ir[j]] = y[ir[j]] * tau + wold[ir[j]] * (1-tau);
		last_seen[ir[j]] = i;
	}
}

/// Finises the "lazy" updates at the end of outer loop
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// iters - number of steps taken in the current outer loop
///			also size of the just finished inner loop
/// d - dimension of the problem
void finish_lazy_updates_ASVRG(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long iters, long d, double *y, double tau)
{
	for (long j = 0; j < d; j++) {
		y[j] -= stepSize * (iters - last_seen[j]) *
			(g[j] + lambda * (w[j] - wold[j]));
		w[j] = y[j] * tau + wold[j] * (1-tau);
	}
}

/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For S2GD algorithm
/// *w - test point; updated in place
/// *g - aggregated gradient
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// d - number of nonzeros of training example
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// i - number of iteration from which this lazy update was called
void update_test_point_sparse_SAG(double *w, double *g, long *last_seen,
	double d, double stepSize, double lambda, mwIndex *ir, long i)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) * (g[ir[j]] + lambda * w[ir[j]]);
		last_seen[ir[j]] = i;
	}
}

void update_test_point_sparse_SAG_plus(double *w, double *g, long *last_seen,
	double d, double stepSize, double lambda, mwIndex *ir, long i, double *cumsteps)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= stepSize * (cumsteps[i] - cumsteps[last_seen[ir[j]]]) * (g[ir[j]] + lambda * w[ir[j]]);
		last_seen[ir[j]] = i;
	}
}


void lazy_update_SGD(double *w, long *last_seen, double stepSize, 
					 double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) * (lambda * w[ir[j]]);
		last_seen[ir[j]] = i;
	}
}

void update_test_point_sparse_SGD(double *x, double *w, double d, double sigmoid,
								  double stepSize, double lambda, mwIndex *ir)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= stepSize * (x[j] * sigmoid);
	}
}