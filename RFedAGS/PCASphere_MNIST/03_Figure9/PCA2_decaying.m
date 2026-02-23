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

% pi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

K = 5; Seed = 1;

% sampling
option = 2;
pi = rand(1, 60);
s = min(ceil(sum(pi)), 60);
stepsize = 2.8e-4;

iidtype = 0;
[fvs1, times1, numgfs1] = PCA2_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);

iidtype = 1;
[fvs2, times2, numgfs2] = PCA2_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);

iidtype = 2;
[fvs3, times3, numgfs3, problem] = PCA2_testing_decaying(iidtype, x0, K, option, s, Seed, stepsize, pi);


%% estimating 


[~, fv8, info8] = rlbfgs(problem, x0);


t = 0:10:2500;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);

times1 = mean(times1, 2);
times2 = mean(times2, 2);
times3 = mean(times3, 2);


options.MarkerIndices = [12, 13, 12, 13, 12, 13];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Excess Risk";
options.legends = ["I.I.D.", 'Non-I.I.D.-slight', 'Non-I.I.D.-heavy'];
options.Markers = {"-^", "->", "-v"};
Ys = [fvs1, fvs2, fvs3] - fv8;

Xs = [t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.number = 1;
options.filePath = 'fig42_iters';
myplot(Xs, Ys, options)


