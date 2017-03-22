%% Prepare the dataset

clear;
mexAll;
addpath('/Users/zhuojiacheng/Google Drive/cookedData/')
% load('protein.tr.mat');
load('ijcnn1.tr.mat')
% load('adult.mat')
% load('rcv1_train.binary.mat')

X = [ones(size(X,1),1) X];
[n, d] = size(X);
X = X';

lambda = 1/n;
Lmax = (0.25 * max(sum(X.^2,1)) + lambda);

% if set to false for not computing function values in process
% if set to true, all algorithms compute function values and display them
history = true;


% % SVRG
% rng(23);

% h = 1 / (10 * Lmax);
% outer_loops = 20;
% m = int64(2*d*ones(outer_loops,1));
% iVals = int64(floor(n*rand(sum(m),1)));

% wSVRG = zeros(d, 1);
% tic;
% if (history)
%     histSVRG = Alg_SVRG(wSVRG, X, y, lambda, h, iVals, m);
% else
%     Alg_SVRG(wSVRG, X, y, lambda, h, iVals, m); 
% end
% t = toc; fprintf('Time spent on SVRG: %f seconds \n', t);


%% SAG
rng(23);

h = 1 / (6 * Lmax);
passes = 40;

iVals = int64(floor(n*rand(passes*n,1)));

wSAG = zeros(d, 1);
tic;
if (history)
    histSAG = Alg_SAG(wSAG, X, y, lambda, h, iVals);
else
    Alg_SAG(wSAG, X, y, lambda, h, iVals);
end
t = toc; fprintf('Time spent on SAG:     %f seconds \n', t);

%% SAG2
rng(23);

h = 1 / (6 * Lmax);
passes = 40;

iVals = int64(floor(n*rand(passes*n,1)));

wSAG2 = zeros(d, 1);
tic;
if (history)
    histSAG2 = Alg_SAGA(wSAG2, X, y, lambda, h, iVals);
else
    SAG2(wSAG2, X, y, lambda, h, iVals);
end
t = toc; fprintf('Time spent on SAGA:     %f seconds \n', t);


%% Plot the results

fstar = 0.2103112055; % for lambda = 1/n; ijcnn1
% fstar = 0.331643965049; % for lambda = 1/n; adult 0.331643965049940
semilogy(histSAG2 - fstar, 'b');
hold on;
semilogy(histSAG - fstar, '--');
