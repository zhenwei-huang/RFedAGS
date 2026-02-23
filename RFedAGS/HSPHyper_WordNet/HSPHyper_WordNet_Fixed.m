set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

features = xlsread('features.xlsx');
emPoin   = xlsread('poincare_points.xlsx');
Y = poin2lor(emPoin');

d = 2;
Manifold = hyperbolicfactory(d);
% Manifold.transp = @paralleltransp;
Manifold.retr = @Manifold.exp;
Manifold.invretr = @Manifold.log;

Dw = 4;

numSamples = max(size(Y)) - 1;

nagents = 9;
ncostterms = ceil(numSamples / nagents);

alpha = zeros(nagents, ncostterms);

fprintf('numSamples:%d, nagents:%d, ncostterms:%d\n', numSamples, nagents, ncostterms)

Z  = {}; % train set of embeddings
WW = {}; % train set of features

for i = 1:numSamples + 1
%     Z{i} = (Y(:,i));
    Z{i} = projection(Y(:,i));
    WW{i} = features(i,:)';
end


indx  = 100; % primate
indxs = [1:indx-1,indx+1:numSamples + 1];
Z  = {Z{indxs}};
WW = {WW{indxs}};
% 'primate' as the test point
w = features(indx,:)';

K = 5;

pi = rand(nagents, 1);

problem.M = Manifold;
prob = cell(nagents,1);
opts = cell(nagents,1);
for i = 1:nagents

    D = {Z{(i-1)*ncostterms+1 : i*ncostterms}};
    W = {WW{(i-1)*ncostterms+1 : i*ncostterms}};    
    alpha(i,:) = computAlpha(w, W);

    probtmp.M = problem.M;
    probtmp.M.transp = problem.M.transp;
    probtmp.M.retr= problem.M.retr;
    probtmp.M.invretr= problem.M.invretr;
    probtmp.ncostterms = ncostterms;
    
    probtmp = ConstructRFLHSPProb_Hyper(D, alpha(i,:), ncostterms, probtmp);
    prob{i} = probtmp;
    
    optstmp.batchsize = ceil(0.5*ncostterms);
    optstmp.maxiter = K;       
    optstmp.pi = pi(i);
    opts{i} = optstmp;                 
end
problem.nagents = nagents;
problem.agent = prob;
problem.cost = @(x) cost(problem, x);
problem.grad = @(x) grad(problem, x);
options.agent = opts;

% checkgradient(problem)

options.verbosity = 2;

options.checkperiod = 10;
options.maxiter = 600;

xtrue = projection(Y(:,indx));

x0 = Manifold.rand();


%% RFedAGSP approximating 
options.option = 2;
options.s = 5;

Seed = 1;

fvs1 = [];
ngfs1 = [];
times1 = [];
dist1 = [];
options.stepsize = 6e-2;
options.localMethod = 'RSGD';
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedAGS\n', seed, K)
    % profile on
    [x1, fv1, info1] = RFedAGSP(problem, x0, options);
    infos1{seed} = info1;
    % profile viewer
    fvs1 = [fvs1, info1.fvs];
    times1 = [times1, info1.CPUtime];

    dist1_ = Manifold.dist(xtrue, x0);
    for i = 1:options.maxiter/options.checkperiod
        dist1_(end+1) = Manifold.dist(xtrue, info1.xs{i});
    end
    dist1_ = reshape(dist1_, length(dist1_), 1);

    dist1 = [dist1, dist1_];
end
figure(1)
t = 0:options.checkperiod:options.maxiter;
gap = ceil(length(t) / 40);
semilogy(t, mean(fvs1,2), '-^', 'LineWidth',2, 'MarkerIndices',1:gap:length(t))

figure(2)
semilogy(mean(times1,2), mean(fvs1,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))

figure(3)
semilogy(0:10:options.maxiter, mean(dist1,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))

%% RFedAvg 

fvs2 = [];
ngfs2 = [];
times2 = [];
dist2 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedAvg\n', seed, K)
    % profile on
    [x2, fv2, info2] = RFedAvg_(problem, x0, options);
    infos2{seed} = info2;
    % profile viewer
    fvs2 = [fvs2, info2.fvs];
    times2 = [times2, info2.CPUtime];

    dist2_ = Manifold.dist(xtrue, x0);
    for i = 1:options.maxiter/options.checkperiod
        dist2_(end+1) = Manifold.dist(xtrue, info2.xs{i});
    end
    dist2_ = reshape(dist2_, length(dist2_), 1);

    dist2 = [dist2, dist2_];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs2,2), '-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))


figure(2)
hold on
semilogy(mean(times2,2), mean(fvs2,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))



figure(3)
hold on
semilogy(0:10:options.maxiter, mean(dist2,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))



%% RFedSVRG

fvs3 = [];
ngfs3 = [];
times3 = [];
dist3 = [];
% options.stepsize = options.stepsize * K;
% K=1;
% for i = 1:nagents
%     options.agent{i}.maxiter = K;
% end
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedSVRG\n', seed, K)
    % profile on
    [x3, fv3, info3] = RFedSVRG_(problem, x0, options);
    infos3{seed} = info3;
    % profile viewer
    fvs3 = [fvs3, info3.fvs];
    times3 = [times3, info3.CPUtime];

    dist3_ = Manifold.dist(xtrue, x0);
    for i = 1:options.maxiter/options.checkperiod
        dist3_(end+1) = Manifold.dist(xtrue, info3.xs{i});
    end
    dist3_ = reshape(dist3_, length(dist3_), 1);

    dist3 = [dist3, dist3_];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs3,2), '-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))
ylabel('Function values')
xlabel('Iterations','FontSize',18)
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'FontSize', 18)

figure(2)
hold on
semilogy(mean(times3,2), mean(fvs3,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))
ylabel('Function values', 'FontSize', 18,'Interpreter','latex')
xlabel('CPU time','FontSize',18, 'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'FontSize', 18)

figure(3)
hold on
semilogy(0:10:options.maxiter, mean(dist3,2),'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))
ylabel('Distance to the true, $\mathrm{dist}(x_t,x_{\mathrm{true}})$', 'FontSize',18, 'Interpreter','latex')
xlabel('Iterations','FontSize',18, 'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'FontSize', 18)


plotball

filename = "workingdata.mat";
save(filename);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = cost(problem, x)
    out = problem.agent{1}.cost(x);
    for i = 2:problem.nagents
        out = out + problem.agent{i}.cost(x);
    end
    out = out / problem.nagents;
end

function out = grad(problem, x)
    out = problem.agent{1}.grad(x);
    for i = 2:problem.nagents
        out = out + problem.agent{i}.grad(x);
    end
    out = out ./ problem.nagents;
end

% computing alpha(w)
function output = computAlpha(w, W)   
    N = length(W);
    Kw = zeros(N, 1);
    K  = zeros(N, N);
    for i = 1:N
        Kw(i) = kernel(w, W{i});
        for j = 1:N
            K(i, j) = kernel(W{i}, W{j});
        end
    end
    output = (K + 1e-5 * eye(N)) \ Kw;
end

function output = kernel(w1, w2)
    dn = norm(w1 - w2, 2);
    sigma = 1/(2*0.03^2);
    output = 1 / exp(sigma * dn * dn);
end

% parallel transport
function output = paralleltransp(x, y, u)
    output = u - (inner_product(y, y, u) / inner_product(x, x, y)) * (x + y);
end

function output = projection(u)
    output = u / sqrt(abs(inner_product(u, u, u)));
%     output = u;
end

function output = inner_product(x, v, u)
    output = v' * u - 2 * v(1) * u(1);
end


function [row, col] = findMinIndex(X, dim)
    if nargin == 1
        dim = 1;
    end
    [n, m] = size(X);
    indx = find(X == min(min(X, [], dim)));
    col = ceil(indx / n);
    row = indx - (col - 1) * n;   
end