set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

m = 200;
r = 5;

Manifold = grassmannfactory(m, r);

x0 = Manifold.rand();


Seed = 10; 

pi = rand(1, 20);
s = ceil(sum(pi));
option = 2;
stepsize = 2e-3;

K1 = 1;
[fvs1, times1, numgfs1, rel_errs1] = LRMCGrass_testing_Syn_fixed(x0, K1, option, s, Seed, stepsize, pi);

K2 = 4;
[fvs2, times2, numgfs2, rel_errs2] = LRMCGrass_testing_Syn_fixed(x0, K2, option, s, Seed, stepsize, pi);

K3 = 8;
[fvs3, times3, numgfs3, rel_errs3] = LRMCGrass_testing_Syn_fixed(x0, K3, option, s, Seed, stepsize, pi);

K4 = 10;
[fvs4, times4, numgfs4, rel_errs4] = LRMCGrass_testing_Syn_fixed(x0, K4, option, s, Seed, stepsize, pi);

K5 = 14;
[fvs5, times5, numgfs5, rel_errs5] = LRMCGrass_testing_Syn_fixed(x0, K5, option, s, Seed, stepsize, pi);

K6 = 18;
[fvs6, times6, numgfs6, rel_errs6, problem] = LRMCGrass_testing_Syn_fixed(x0, K6, option, s, Seed, stepsize, pi);

[x7, fv7, info7] = rlbfgs(problem, x0);

t = 0:1:100;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);
fvs4 = mean(fvs4, 2);
fvs5 = mean(fvs5, 2);
fvs6 = mean(fvs6, 2);

rel_errs1 = mean(rel_errs1, 2);
rel_errs2 = mean(rel_errs2, 2);
rel_errs3 = mean(rel_errs3, 2);
rel_errs4 = mean(rel_errs4, 2);
rel_errs5 = mean(rel_errs5, 2);
rel_errs6 = mean(rel_errs6, 2);

options.MarkerIndices = [5, 5, 5, 5, 5, 5, 5];
options.LineWidth = 1.8;
options.Markersize = 10;
options.legends = ["K="+string(K1), "K="+string(K2), "K="+string(K3),...
                   "K="+string(K4), "K="+string(K5), "K="+string(K6)];
options.Markers = {"-^", "->", "-v","-<", "-.+", "-.d", "-.h"};
% options.Markers = {"-", "-", "-","-.", "-.", "-."};
Ys = [fvs1, fvs2, fvs3, fvs4, fvs5, fvs6] - fv7;

Xs = [t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.ylabel = "Optimality gap";
options.number = 1;
options.filePath = 'fig7_iters';
myplot(Xs, Ys, options)


Ys = [rel_errs1, rel_errs2, rel_errs3, rel_errs4, rel_errs5, rel_errs6];
options.xlabel = "Iterations";
options.ylabel = "Relative Errors";
options.number = 2;
options.filePath = 'fig7_rel_errs';
myplot(Xs, Ys, options)

