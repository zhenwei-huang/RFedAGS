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

pi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

K = 5; Seed = 10;

% sampling
option = 3;
stepsize = 3.5e-4;

rho1 = 0.3;
s = 60 * rho1;
[fvs1, times1, numgfs1] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);

rho2 = 0.5;
s = 60 * rho2;
[fvs2, times2, numgfs2] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);

rho3 = 0.7;
s = 60 * rho3;
[fvs3, times3, numgfs3] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);


%% estimating 
option = 2;
% stepsize = 8e-5;

pi = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
[fvs5, times5, numgfs5] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);

pi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
[fvs6, times6, numgfs6] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);

pi = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
[fvs7, times7, numgfs7, problem] = PCA1_testing_decaying(x0, K, option, s, Seed, stepsize, pi);

[~, fv8, info8] = rlbfgs(problem, x0);


t = 0:10:1000;
fvs1 = mean(fvs1, 2);
fvs2 = mean(fvs2, 2);
fvs3 = mean(fvs3, 2);
fvs5 = mean(fvs5, 2);
fvs6 = mean(fvs6, 2);
fvs7 = mean(fvs7, 2);

times1 = mean(times1, 2);
times2 = mean(times2, 2);
times3 = mean(times3, 2);
times5 = mean(times5, 2);
times6 = mean(times6, 2);
times7 = mean(times7, 2);


options.MarkerIndices = [5, 4, 5, 4, 5, 4];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Excess Risk";
options.legends = ["Scheme I-"+string(rho1), "Scheme I-"+string(rho2), "Scheme I-"+string(rho3),...
                   "Scheme II-"+string(rho1),"Scheme II-"+string(rho2),"Scheme II-"+string(rho3)];
options.Markers = {"-^", "->", "-v","-.o", "-.+", "-.d"};
% options.Markers = {"-", "-", "-","-.", "-.", "-."};
Ys = [fvs1, fvs2, fvs3, fvs5, fvs6, fvs7] - fv8;

Xs = [t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.number = 1;
options.filePath = 'fig2_iters';
myplot(Xs, Ys, options)


Xs = [times1, times2, times3, times5, times6, times7];
options.xlabel = "CPU time (s)";
options.number = 2;
options.filePath = 'fig2_ER';
myplot(Xs, Ys, options)

