%% Prepare the dataset

clear;
mex Alg_SVRG.cpp -largeArrayDims;
mex Alg_SVRG2.cpp -largeArrayDims;
mex Alg_ASVRG.cpp -largeArrayDims;
mex Alg_Katyusha.cpp -largeArrayDims;

addpath('cookedData/')
% load('protein.tr.mat');
load('ijcnn1.tr.mat')
% load('adult.mat')
% load('rcv1_train.binary.mat')

% X = [ones(size(X,1),1) X];
[n, d] = size(X);
X = full(X');
% X = X';

lambda = 1/10000;
Lmax = (0.25 * max(sum(X.^2,1)) + lambda);

% if set to false for not computing function values in process
% if set to true, all algorithms compute function values and display them
history = true;


%% SVRG
rng(23);

h = 1 / (10 * Lmax);
outer_loops = 20;
m = int64(2*n*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

wSVRG = zeros(d, 1);
tic;
if (history)
    histSVRG = Alg_SVRG(wSVRG, X, y, lambda, h, iVals, m);
else
    Alg_SVRG(wSVRG, X, y, lambda, h, iVals, m); 
end
t = toc; fprintf('Time spent on SVRG: %f seconds \n', t);

%% SVRG2
rng(23);

h = 1 / (10 * Lmax);
outer_loops = 20;
m = int64(2*n*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

wSVRG2 = zeros(d, 1);
tic;
if (history)
    histSVRG2 = Alg_SVRG2(wSVRG2, X, y, lambda, h, iVals, m);
else
    Alg_SVRG2(wSVRG2, X, y, lambda, h, iVals, m); 
end
t = toc; fprintf('Time spent on SVRG: %f seconds \n', t);

%% ASVRG    
rng(23);

w_ASVRG_d = zeros(d, 1);
%     tau  = 1 - Lmax*lambda/(1-Lmax*lambda);
alpha = 1 / (3*Lmax);
tau = 0.85;

outer_loops = 20;
m = int64(2*n*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

tic;
if (history)
    histASVRG_d = Alg_ASVRG(w_ASVRG_d, full(X), y, lambda, alpha, iVals, m, tau);
else
    Alg_ASVRG(w_ASVRG_d, X, y, lambda, alpha, tau, iVals, m, tau);
end
time_ASVRG = toc;
fprintf('Time spent on ASVRG: %f seconds \n', time_ASVRG);
xASVRG_l2 = 0:outer_loops;

%% Katyusha
rng(23);

w_Katyusha = zeros(d, 1);
%     tau  = 1 - Lmax*lambda/(1-Lmax*lambda);
alpha = 1 / (3*tau*Lmax);
tau = min(0.5, sqrt(2*n*lambda/(3*Lmax)));

outer_loops = 20;
m = int64(2*n*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

tic;
if (history)
    histKatyusha = Alg_Katyusha(w_Katyusha, full(X), y, lambda, alpha, iVals, m, tau);
else
    Alg_Katyusha(w_Katyusha, X, y, lambda, alpha, tau, iVals, m, tau);
end
time_Katyusha = toc;
fprintf('Time spent on ASVRG: %f seconds \n', time_Katyusha);
xKatyusha_l2 = 0:outer_loops;

%% Plot the results

fstar = 0.23641172058; % for lambda = 1/n; ijcnn1
% fstar = 0.2364117205;
% fstar = 0.343943296559; % for lambda = 1/n; adult 0.331643965049940
% fstar = 0.31984097907; % protein
semilogy(histSVRG2 - fstar, 'b');
hold on;
semilogy(histSVRG - fstar, 'r');
hold on;
semilogy(histASVRG_d - fstar, 'g');
hold on;
semilogy(histKatyusha - fstar, 'y');
% hold on;
% semilogy(histKatyusha - fstar, 'g');
