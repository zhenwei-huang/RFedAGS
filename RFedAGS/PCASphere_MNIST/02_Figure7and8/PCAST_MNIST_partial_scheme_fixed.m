set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

% load('MNIST_training_50_100.mat')
% load('MNIST_training_50_100000.mat')
% load('MNIST_train_60_iid_.mat')
% load('MNIST_train_60_noniid_small.mat')
% load('MNIST_train_60_noniid_large.mat')

d = 784;
Manifold = spherefactory(d, 1);
Manifold.retr = @Manifold.exp;
Manifold.invretr = @Manifold.log;
Manifold.transp = @Manifold.isotransp;
problem.M = Manifold;


K1 = 5;
%%% construct agents
% pi = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,...
%       0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
%       0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,...
%       0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];

pi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,...
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];


options.checkperiod = 10;
options.maxiter = 500;
rng(1)
x0 = Manifold.rand();

% for step size
% options.stepsize_type = 'decay';
% options.decaysteps = 10;
options.stepsize_type = 'fix';

load('MNIST_train_60_noniid_small.mat')
nagents = double(nagents);
ncostterms = double(ncostterms);
options.verbosity = 2;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);

[x7, fv7, info7] = rlbfgs(problem, x0);

%% full participation
options.option = 1;  
options.s = nagents / 2;

Seed = 1;
%%%%%% SGD %%%%%%%
fvs1 = [];
ngfs1 = [];
numgfs1 = [];
times1 = [];
options.stepsize = 8e-5;
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
figure(1)
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs1,2)-fv7, '-^', 'LineWidth',2)

figure(2) 
semilogy(mean(times1,2), mean(fvs1,2) - fv7,'-^', 'LineWidth',2)

%% sampling 
options.option = 3;  
options.s = nagents / 2;

Seed = 1;
%%%%%% SGD %%%%%%%
fvs3 = [];
ngfs3 = [];
times3 = [];
% options.stepsize = 3e-4;
options.localMethod = 'RSGD';
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x3, fv3, info3] = RFedAGSP(problem, x0, options);
    fvs3 = [fvs3, info3.fvs];
    ngfs3 = [ngfs3, info3.ngfs];
    times3 = [times3, info3.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs3,2) - fv7, '->', 'LineWidth',2)

figure(2) 
hold on
semilogy(mean(times3,2), mean(fvs3,2) - fv7,'->', 'LineWidth',2)


%% approximating
options.option = 2;
%%%%%% SGD %%%%%%%
fvs5 = [];
ngfs5 = [];
times5 = [];
% options.stepsize = 3e-5;
options.localMethod = 'RSGD';
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x5, fv5, info5] = RFedAGSP(problem, x0, options);
    fvs5 = [fvs5, info5.fvs];
    ngfs5 = [ngfs5, info5.ngfs];
    times5 = [times5, info5.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t,mean(fvs5,2)-fv7, '-v', 'LineWidth',2)
ylabel('Excess Risk', 'FontSize',18)
xlabel('Iterations','FontSize',18)
legend('RFedAGS-full', 'RFedAGS-sampling', 'RFedAGS-estimating', 'FontSize', 18)


figure(2) 
hold on
semilogy(mean(times5,2),mean(fvs5,2)-fv7,'-v', 'LineWidth',2)
ylabel('Excess Risk', 'FontSize',18)
xlabel('CPU time (s)','FontSize',18)
legend('RFedAGS-full', 'RFedAGS-sampling', 'RFedAGS-estimating', 'FontSize', 18)



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


