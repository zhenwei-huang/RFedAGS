set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

d = 50;
r = 1;
Manifold = stiefelfactory(d, r);
problem.M = Manifold;

K1 = 50;
%%% construct agents

pi = rand(1, 60);

options.checkperiod = 10;
options.maxiter = 150;
rng(1)
x0 = Manifold.rand();

% for step size
% options.stepsize_type = 'decay';
% options.decaysteps = 10;
options.stepsize_type = 'fix';

filename = "PCA_Stief_Syn_train_AGSvsTM";

% filename = 'MNIST_train_60_noniid_small';
% load(filename + ".mat")
% nagents = double(nagents);
% ncostterms = double(ncostterms);
nagents = 2;
ncostterms = 60;

data = zeros(nagents * ncostterms, d);
data(1 : ncostterms, :) = (1 / nagents) * randn(ncostterms, d); 
data(ncostterms + 1 : 2 * ncostterms, :) = (800 / nagents) * randn(ncostterms, d); 
% data(2*ncostterms + 1 : 3 * ncostterms, :) = (100 / nagents) * randn(ncostterms, d); 
options.verbosity = 2;


data = data(1:ncostterms * nagents, :);

[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ncostterms, K1, pi);

[x7, fv7, info7] = rlbfgs(problem, x0,options);

%% RFedAGSP full participation
options.option = 1;

Seed = 1;

fvs1 = [];
ngfs1 = [];
numgfs1 = [];
times1 = [];
options.stepsize = .1e-6;
options.localMethod = 'RSGD';
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    % profile on
    [x1, fv1, info1] = RFedAGSP(problem, x0, options);
    % profile viewer
    fvs1 = [fvs1, info1.fvs];
    numgfs1 = [numgfs1, info1.numgfs];
    times1 = [times1, info1.CPUtime];
end
gap = ceil(options.maxiter / options.checkperiod / 30);
figure(1)
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs1,2)-fv7, '-^', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% RFedAvg
fvs5 = [];
ngfs5 = [];
times5 = [];
options.localMethod = 'RSGD';
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ncostterms, K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x5, fv5, info5] = RFedAvg_(problem, x0, options);
    fvs5 = [fvs5, info5.fvs];
    ngfs5 = [ngfs5, info5.ngfs];
    times5 = [times5, info5.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t,mean(fvs5,2)-fv7, '-o', 'LineWidth', 2, 'MarkerIndices', 1:gap:(options.maxiter / options.checkperiod))
ylabel('Optimality gap', 'FontSize',18,'Interpreter','latex')
xlabel('Iterations','FontSize',18,'Interpreter','latex')
tit = '$(r,d,N,S)=(' + string(r) + ',' + string(d) + ',' + string(nagents) + ',' + string(ncostterms) + ')$';
title(tit,'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'FontSize', 18)

% saveas(gcf, filename + "_ER.fig")



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [problem, options] = construct_prob(D, problem, options, nagents, ncostterms, batchsize, K, pi)
    
    prob = cell(nagents,1);
    opts = cell(nagents,1);
    for i = 1:nagents
        Z = cell(ncostterms,1);
        for j = 1:ncostterms
            Z{j} = D((i-1)*ncostterms + j, :)';
        end 
        probtmp.M = problem.M;
        probtmp.ncostterms = ncostterms;
        probtmp = ConstructRFLPCAProb_St_(Z, ncostterms, probtmp);
        prob{i} = probtmp;
        
        optstmp.batchsize = batchsize;
        optstmp.maxiter = K;       
        optstmp.pi = pi(i);
        optstmp.mu = 1e-5;
        opts{i} = optstmp;                 
    end
    problem.nagents = nagents;
    problem.agent = prob;
    problem.cost = @(x) cost(problem, x);
    problem.grad = @(x) grad(problem, x);
    options.agent = opts;
end


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


