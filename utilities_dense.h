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
//// This contains functions for dense data matrix                ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/// Compute the function value of average regularized logistic loss
/// *w - tesp point
/// *Xt - data matrix
/// *y - set of labels
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
double compute_function_value(double* w, double *Xt, double *y,
	long n, long d, double lambda)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = 0; j < d; j++) {
			tmp += Xt[i*d + j] * w[j];
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

/// Update the test point *w in place
/// Hint: Use this function only when you assume the gradient is fully dense
/// *w - test point; updated in place
/// *g - gradient (update direction)
/// d - dimension of the problem
/// stepSize - step-size parameter
void update_test_point_dense(double *w, double *g, long d, double stepSize)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (g[j]);
	}
}

/// Update the test point *w in place once you have everything prepared
/// *x - training example
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *gold - full gradient computed at point *wold
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// d - dimension of the problem
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
void update_test_point_dense_S2GD(double *x, double *w, double *wold, 
	double *gold, double sigmoid, double sigmoidold,
	long d, double stepSize, double lambda)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (gold[j] + x[j] *
			(sigmoid - sigmoidold) + lambda * (w[j] - wold[j]));
	}
}

/// Update the test point *w in place once you have everything prepared
/// *w - test point; updated in place
/// *g - aggregated gradient
/// d - dimension of the problem
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
void update_test_point_dense_SAG(double *w, double *g, long d, double stepSize, double lambda)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (g[j] + lambda * w[j]);
	}
}

/// Update the test point *w in place once you have everything prepared
/// *x - training example
/// *w - test point; updated in place
/// d - dimension of the problem
/// sigmoid - scalar value of derivative of loss
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
void update_test_point_dense_SGD(double *x, double *w, long d, 
				double sigmoid, double stepSize, double lambda)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (x[j] * sigmoid + lambda * w[j]);
	}
}
