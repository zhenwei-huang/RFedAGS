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
stepsize = 8e-5;

rho1 = 0.3;
s = 60 * rho1;
[fvs1, times1, numgfs1] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);

% rho2 = 0.5;
% s = 60 * rho2;
% [fvs2, times2, numgfs2] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);
% 
% rho3 = 0.7;
% s = 60 * rho3;
% [fvs3, times3, numgfs3] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);


%% true
option = 4;

pi = [0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,...
      0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
[fvs5, times5, numgfs5] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);


%% estimating
option = 2;
% pi = [0.05, 0.05, 0.05, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55,...
%       0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55,...
%       0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
pi = [0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,...
      0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,...
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
[fvs6, times6, numgfs6, problem] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);

% pi = [0.05, 0.05, 0.05, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,...
%       0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,...
%       0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,...
%       0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,...
%       0.75, 0.75, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,...
%       0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
% [fvs7, times7, numgfs7, problem] = PCA1_testing_fixed_lazy(x0, K, option, s, Seed, stepsize, pi);


[~, fv8, info8] = rlbfgs(problem, x0);


t = 0:10:400;
fvs1 = mean(fvs1, 2);
% fvs2 = mean(fvs2, 2);
% fvs3 = mean(fvs3, 2);
fvs5 = mean(fvs5, 2);
fvs6 = mean(fvs6, 2);
% fvs7 = mean(fvs7, 2);

times1 = mean(times1, 2);
% times2 = mean(times2, 2);
% times3 = mean(times3, 2);
times5 = mean(times5, 2);
times6 = mean(times6, 2);
% times7 = mean(times7, 2);


options.MarkerIndices = [2, 3, 2, 3, 2, 3];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Excess Risk";
% options.legends = ["Scheme I-"+string(rho1), "Scheme I-"+string(rho2),...
%                    "Scheme I-"+string(rho3), "Scheme II-"+string(rho1),...
%                    "Scheme II-"+string(rho2), "Scheme II-"+string(rho3)];
options.legends = ["Scheme I-"+string(rho1), "Scheme II-True-"+string(rho1),"Scheme II-Freq-"+string(rho1)];

options.Markers = {"-^", "->", "-v","-<", "-.+", "-.d"};
% Ys = [fvs1, fvs2, fvs3, fvs5, fvs6, fvs7] - fv8;
Ys = [fvs1, fvs5, fvs6] - fv8;


% Xs = [times1, times2, times3, times5, times6, times7];
Xs = [times1, times5, times6];
options.xlabel = "CPU time (s)";
options.number = 2;
options.filePath = 'fig31_ER';
myplot(Xs, Ys, options)

