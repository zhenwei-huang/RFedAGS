set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

d = 784;
Manifold = spherefactory(d, 1);
Manifold.retr = @Manifold.exp;
Manifold.invretr = @Manifold.log;
Manifold.transp = @Manifold.isotransp;

x0 = Manifold.rand();

Seed = 10;

% estimating
option = 2;
stepsize = 8e-5;
iidtype = 2;
pi = rand(1, 60);
s = min(ceil(sum(pi)), 60);


K1 = 2;
[fvs1, times1, numgfs1] = PCA3_testing_fixed(iidtype, x0, K1, option, s, Seed, stepsize, pi);

K2 = 4;
% stepsize = 8e-5 * (2/5);
[fvs2, times2, numgfs2] = PCA3_testing_fixed(iidtype, x0, K2, option, s, Seed, stepsize, pi);

K3 = 6; 
% stepsize = 8e-5 * (2 / 10);
[fvs3, times3, numgfs3] = PCA3_testing_fixed(iidtype, x0, K3, option, s, Seed, stepsize, pi);

K4 = 8;
[fvs4, times4, numgfs4] = PCA3_testing_fixed(iidtype, x0, K4, option, s, Seed, stepsize, pi);

K5 = 10; 
[fvs5, times5, numgfs5] = PCA3_testing_fixed(iidtype, x0, K5, option, s, Seed, stepsize, pi);

K6 = 14; 
[fvs6, times6, numgfs6] = PCA3_testing_fixed(iidtype, x0, K6, option, s, Seed, stepsize, pi);

K7 = 18; 
[fvs7, times7, numgfs7, problem] = PCA3_testing_fixed(iidtype, x0, K7, option, s, Seed, stepsize, pi);

[~, fv8, info8] = rlbfgs(problem, x0);


t = 0:10:900;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);
fvs4 = mean(fvs4, 2);
fvs5 = mean(fvs5, 2);
fvs6 = mean(fvs6, 2);
fvs7 = mean(fvs7, 2);


options.MarkerIndices = [3, 2, 3, 2, 3, 2, 2, 2];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Optimality gap";
options.legends = ["K=2", "K=4", "K=6", "K=8", "K=10", "K=14", "K=18"];
options.Markers = {"-^", "->", "-v","-<","-o","-d", "-h"};
Ys = [fvs1, fvs2, fvs3, fvs4, fvs5, fvs6, fvs7] - fv8;

Xs = [t', t', t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.number = 1;
options.filePath = 'fig51_iters';
myplot(Xs, Ys, options)
