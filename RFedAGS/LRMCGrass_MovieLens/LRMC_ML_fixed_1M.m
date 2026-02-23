% set(0,'defaultaxesfontsize',22, ...
%    'defaultaxeslinewidth',0.7, ...
%    'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
% set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

m = 3952;
r = 6;

Manifold = grassmannfactory(m, r);

x0 = Manifold.rand();
% I = eye(m);
% x0 = I(:,1:r);


Seed = 1; 

lambda = 1e-2;
pi = rand(1, 40);
s = ceil(sum(pi));
option = 2;
stepsize = 1e-3;

K1 = 1;
[fvs1, times1, numgfs1, rmses1] = LRMCGrass_testing_ML_fixed(x0, K1, option, r, lambda, Seed, stepsize, pi);

K2 = 4;
[fvs2, times2, numgfs2, rmses2] = LRMCGrass_testing_ML_fixed(x0, K2, option, r, lambda, Seed, stepsize, pi);

K3 = 8;
[fvs3, times3, numgfs3, rmses3] = LRMCGrass_testing_ML_fixed(x0, K3, option, r, lambda, Seed, stepsize, pi);

K4 = 10;
[fvs4, times4, numgfs4, rmses4, problem] = LRMCGrass_testing_ML_fixed(x0, K4, option, r, lambda, Seed, stepsize, pi);

options.maxiter = 40;
[x7, fv7, info7] = rlbfgs(problem, x0,options);

t = 0:1:options.maxiter;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);
fvs4 = mean(fvs4, 2);

rmses1 = mean(rmses1, 2);
rmses2 = mean(rmses2, 2);
rmses3 = mean(rmses3, 2);
rmses4 = mean(rmses4, 2);

options.MarkerIndices = [5, 5, 5, 5, 5, 5];
options.LineWidth = 1.8;
options.Markersize = 10;
options.legends = ["K="+string(K1), "K="+string(K2), "K="+string(K3),...
                   "K="+string(K4)];
options.Markers = {"-^", "->", "-v","-<", "-.+", "-.d"};
% options.Markers = {"-", "-", "-","-.", "-.", "-."};
Ys = [fvs1, fvs2, fvs3, fvs4] - fv7;

Xs = [t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.ylabel = "Excess Risk";
options.number = 1;
options.filePath = 'fig8_iters';
myplot(Xs, Ys, options)


Ys = [rmses1, rmses2, rmses3, rmses4];
options.xlabel = "Iterations";
options.ylabel = "RMSE on testing dataset";
options.number = 2;
options.filePath = 'fig8_rmse';
myplot(Xs, Ys, options)

