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
option = 2;
stepsize = 8e-5;
iidtype = 1;
pi1 = rand(1, 60);
pi2 = 0.8*pi1;
pi3 = 0.6*pi1;
s = min(ceil(sum(pi)), 60);

K1 = 5;
method = "RFedAGS";
[fvs11, times11, numgfs11] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi1);
[fvs12, times12, numgfs12] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi2);
[fvs13, times13, numgfs13] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi3);

method = "RFedAvg";
[fvs21, times21, numgfs21] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi1);
[fvs22, times22, numgfs22] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi2);
[fvs23, times23, numgfs23] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi3);

method = "RFedSVRG";
[fvs31, times31, numgfs31] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi1);
[fvs32, times32, numgfs32] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi2);
[fvs33, times33, numgfs33] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi3);

method = "RFedProj";
[fvs41, times41, numgfs41] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi1);
[fvs42, times42, numgfs42] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi2);
[fvs43, times43, numgfs43] = PCA5_testing_fixed(iidtype, x0, K1, method, option, s, Seed, stepsize, pi3);

[~, fv8, info8] = rlbfgs(problem, x0);

t = 0:10:600;
fvs11 = mean(fvs11, 2);
fvs12 = mean(fvs12, 2);
fvs13 = mean(fvs13, 2);

fvs21 = mean(fvs21, 2);
fvs22 = mean(fvs22, 2);
fvs23 = mean(fvs23, 2);

fvs31 = mean(fvs31, 2);
fvs32 = mean(fvs32, 2);
fvs33 = mean(fvs33, 2);

fvs41 = mean(fvs41, 2);
fvs42 = mean(fvs42, 2);
fvs43 = mean(fvs43, 2);

options.MarkerIndices = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2];
options.LineWidth = 1.8;
options.Markersize = 10;
options.ylabel = "Optimality gap";
options.legends = ["RFedAGS-0.5", "RFedAGS-0.4", "RFedAGS-0.3", ...
                   "RFedAvg-0.5", "RFedAvg-0.4", "RFedAvg-0.3", ...
                   "RFedSVRG-0.5", "RFedSVRG-0.4", "RFedSVRG-0.3", ...
                   "RFedProj-0.5", "RFedProj-0.4", "RFedProj-0.3"];
options.Markers = {"-^", "-.^", "--^", ...
                   "->", "-.>", "-->", ...
                   "-v", "-.v", "--v", ...
                   "-<", "-.<", "--<"};
Ys = [fvs11, fvs12, fvs13, fvs21, fvs22, fvs23, fvs31, fvs32, fvs33, fvs41, fvs42, fvs43] - fv8;

Xs = [t', t', t', t', t', t', t', t', t', t', t', t'];
options.xlabel = "Iterations";
options.number = 1;
options.filePath = 'fig62_iters';
myplot(Xs, Ys, options)


times11 = mean(times11, 2);
times12 = mean(times12, 2);
times13 = mean(times13, 2);

times21 = mean(times21, 2);
times22 = mean(times22, 2);
times23 = mean(times23, 2);

times31 = mean(times31, 2);
times32 = mean(times32, 2);
times33 = mean(times33, 2);

times41 = mean(times41, 2);
times42 = mean(times42, 2);
times43 = mean(times43, 2);

Xs = [times11, times12, times13, times21, times22, times33, times31, times32, times33, times41, times42, times43];

options.xlabel = "CPU time (s)";
options.number = 2;
options.filePath = 'fig62_time';
myplot(Xs, Ys, options)


