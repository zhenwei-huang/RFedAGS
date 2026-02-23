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


Seed = 1;

% estimating
% stepsize = 8e-5;
stepsize = 3.5e-4;

iidtype = 1;
pi = rand(1, 60);
s = min(ceil(sum(pi)), 60);

K = 5;
option = 4;
[fvs1, times1, numgfs1, ~, fvs1_] = PCA41_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);

option = 2;
[fvs2, times2, numgfs2, ~, fvs2_] = PCA41_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);

option = 4;
[fvs3, times3, numgfs3, problem, fvs3_, problem_] = PCA43_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);

[~, fv8, info8] = rlbfgs(problem, x0);

[~, fv9, info9] = rlbfgs(problem_, x0);


t = 0:10:1000;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);

fvs1_ = mean(fvs1_, 2);
fvs2_ = mean(fvs2_, 2);
fvs3_ = mean(fvs3_, 2);


options.MarkerIndices = [3, 2, 3, 2, 3, 2];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Excess Risk";
options.legends = ["True", "Approximating", "Biased"];
options.Markers = {"-^", "->", "-v"};
Xs = [t', t', t'];

Ys = [fvs1, fvs2, fvs3] - fv8;
options.xlabel = "Iterations";
options.number = 1;
options.filePath = 'fig02_iters';
myplot(Xs, Ys, options)


Ys = [fvs1_, fvs2_, fvs3_] - fv9;
options.number = 2;
options.filePath = 'fig02_iters_';
myplot(Xs, Ys, options)

