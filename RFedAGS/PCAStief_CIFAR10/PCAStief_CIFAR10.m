set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

d = 3072;
% d = 784;
r = 4;
Manifold = stiefelfactory(d, r);
% Manifold = spherefactory(d);
problem.M = Manifold;


K1 = 5;
%%% construct agents
% pi = [0.2 * ones(1,10);
%       0.3 * ones(1,10);
%       0.4 * ones(1,10);
%       0.5 * ones(1,10);
%       0.6 * ones(1,10);
%       0.7 * ones(1,10)];
pi = rand(1, 60);

options.checkperiod = 10;
options.maxiter = 600;
rng(1)
x0 = Manifold.rand();

% for step size
% options.stepsize_type = 'decay';
% options.decaysteps = 10;
options.stepsize_type = 'fix';

filename = "CIFAR10_train_50_noniid_slight";

load(filename + ".mat")

nagents = double(nagents);
ncostterms = double(ncostterms);

options.verbosity = 2;

nagents = 50;
ncostterms = 1000;

% data_ = [];
% for i = 1:50
%     data_ = [data_;data((i-1)*1000+1 : (i-1)*1000 + ncostterms, :)];
% end
% data = data_;
% data = data(1:ncostterms * nagents, :);

[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);

[x7, fv7, info7] = rlbfgs(problem, x0,options);

%% RFedAGSP approximating 
options.option = 2;

Seed = 2;

fvs1 = [];
ngfs1 = [];
numgfs1 = [];
times1 = [];
options.stepsize = 3e-5;
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
gap = ceil(options.maxiter / options.checkperiod / 30);
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs1,2)-fv7, '-^', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
semilogy(mean(times1,2), mean(fvs1,2) - fv7,'-^', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% RFedAvg
options.option = 2;

fvs2 = [];
ngfs2 = [];
times2 = [];
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x2, fv2, info2] = RFedAvg_(problem, x0, options);
    fvs2 = [fvs2, info2.fvs];
    ngfs2 = [ngfs2, info2.ngfs];
    times2 = [times2, info2.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs2, 2) - fv7, '->', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times2,2), mean(fvs2,2) - fv7,'->', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))


%% RFedSVRG
options.option = 2;

fvs3 = [];
ngfs3 = [];
times3 = [];
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x3, fv3, info3] = RFedSVRG_(problem, x0, options);
    fvs3 = [fvs3, info3.fvs];
    ngfs3 = [ngfs3, info3.ngfs];
    times3 = [times3, info3.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs3,2) - fv7, '-v', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times3,2), mean(fvs3,2) - fv7,'-v', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% RFedProj
fvs4 = [];
ngfs4 = [];
times4 = [];
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x4, fv4, info4] = RFedProj_(problem, x0, options);
    fvs4 = [fvs4, info4.fvs];
    ngfs4 = [ngfs4, info4.ngfs];
    times4 = [times4, info4.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs4,2) - fv7, '-<', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times4,2), mean(fvs4,2) - fv7,'-<', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% ZO-RFedProj
fvs5 = [];
ngfs5 = [];
times5 = [];
options.localMethod = 'RSGD';
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
for seed = 1:Seed
    fprintf('seed:%d, K:%d\n', seed, K1)
    [x5, fv5, info5] = ZORFedProj_(problem, x0, options);
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
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'RFedProj', 'ZO-RFedProj', 'FontSize', 18)

saveas(gcf, filename + "_ER.fig")

figure(2) 
hold on
semilogy(mean(times5,2),mean(fvs5,2)-fv7,'-o', 'LineWidth', 2, 'MarkerIndices', 1:gap:(options.maxiter / options.checkperiod))
ylabel('Optimality gap', 'FontSize',18,'Interpreter','latex')
xlabel('CPU time (s)','FontSize',18,'Interpreter','latex')
title(tit,'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'RFedProj', 'ZO-RFedProj', 'FontSize', 18)

saveas(gcf, filename + "_Time.fig") 


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
        optstmp.mu = 1e-5;
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


