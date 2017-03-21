%% Prepare the dataset

clear;
mexAll;
addpath('/Users/zhuojiacheng/Google Drive/cookedData/')
% load('protein.tr.mat');
% load('ijcnn1.tr.mat')
load('adult.mat')

X = [ones(size(X,1),1) X];
[n, d] = size(X);
X = full(X');

lambda = 1/n;
Lmax = (0.25 * max(sum(X.^2,1)) + lambda);

% if set to false for not computing function values in process
% if set to true, all algorithms compute function values and display them
history = true;


rng(23);

h = 1 / (10 * Lmax);
outer_loops = 20;
m = int64(2*d*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

w = zeros(d, 1);
tic;
if (history)
    histS2GD = Alg_SVRG(w, X, y, lambda, h, iVals, m);
else
    Alg_SVRG(w, X, y, lambda, h, iVals, m); 
end
t = toc; fprintf('Time spent on S2GDcon: %f seconds \n', t);


rng(23)

h = 1 / (10 * Lmax);
outer_loops = 20;
m = int64(2*d*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

w = zeros(d, 1);
tic;
if (history)
    histSVRG3 = SVRG3(w, X, y, lambda, h, iVals, m);
else
    SVRG3(w, X, y, lambda, h, iVals, m);
end
t = toc; fprintf('Time spent on S2GD:   %f seconds \n', t);



%% Plot the results

fstar = 0.31388; % for lambda = 1/n;
semilogy(histS2GD - fstar, 'b');
hold on;
semilogy(histSVRG3 - fstar, 'r');
