%% Prepare the dataset

clear;
mexAll;
addpath('cookedData/')
% load('protein.tr.mat');
load('ijcnn1.tr.mat')
% load('adult.mat')
% load('rcv1_train.binary.mat')

% X = [ones(size(X,1),1) X];
[n, d] = size(X);
% X = full(X');
X = X';

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


%% ASVRG    
rng(23);

w_ASVRG = zeros(d, 1);
%     tau  = 1 - Lmax*lambda/(1-Lmax*lambda);
alpha = 1 / (3*Lmax);
tau = 0.85;

outer_loops = 20;
m = int64(2*n*ones(outer_loops,1));
iVals = int64(floor(n*rand(sum(m),1)));

tic;
if (history)
    histASVRG = Alg_ASVRG(w_ASVRG, X, y, lambda, alpha, iVals, m, tau);
else
    Alg_ASVRG(w_ASVRG, X, y, lambda, alpha, tau, iVals, m, tau);
end
time_ASVRG = toc;
fprintf('Time spent on ASVRG: %f seconds \n', time_ASVRG);
xASVRG_l2 = 0:outer_loops;


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

%% Plot the results

fstar = 0.2103112055; % for lambda = 1/n; ijcnn1
fstar = 0.2364117205;
% fstar = 0.331643965049; % for lambda = 1/n; adult 0.331643965049940
% fstar = 0.3137; % protein
semilogy(histSAG2 - fstar, 'b');
hold on;
semilogy(histSAG - fstar, '--');
hold on;
semilogy(histSVRG - fstar, 'r');
hold on;
semilogy(histASVRG - fstar, 'g');
hold on;
semilogy(histASVRG_d - fstar, 'y');
% hold on;
% semilogy(histKatyusha - fstar, 'g');
