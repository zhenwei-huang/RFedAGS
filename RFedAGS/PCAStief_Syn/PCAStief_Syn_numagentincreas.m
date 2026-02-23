set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

d = 100;
r = 5;
Manifold = stiefelfactory(d, r);
problem.M = Manifold;

K1 = 5;
%%% construct agents

data = zeros(200000, d);
for i = 1:200
    datatmp = (i / 200) * randn(500, d);
    data((i-1)*500 + 1 : i*500, :) = datatmp;
end

pi = rand(1, 200);

options.checkperiod = 10;
options.maxiter = 2500;
rng(1)
x0 = Manifold.rand();

% for step size
% options.stepsize_type = 'decay';
% options.decaysteps = 10;
options.stepsize_type = 'fix';

filename = "PCA_Stief_Syn_train_numagentincreas";

options.verbosity = 2;

Seed = 4;

%% N = 60, S = 2000 
nagents = 60; ncostterms = 1000;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
[x1_, fv1_, info1_] = rlbfgs(problem, x0,options);

options.option = 2;

fvs1 = [];
ngfs1 = [];
numgfs1 = [];
times1 = [];
options.stepsize = 2.e-2;
options.localMethod = 'RSGD';
for seed = 1:Seed
    fprintf('seed:%d, K:%d, nagents:%d\n', seed, K1, nagents);
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
semilogy(t, mean(fvs1,2) - fv1_, '-^', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))


figure(2) 
semilogy(mean(times1,2), mean(fvs1,2) - fv1_,'-^', 'LineWidth',2, 'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% N = 80, S = 1000 
options.option = 2;
nagents = 80; ncostterms = 1000;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
[x2_, fv2_, info2_] = rlbfgs(problem, x0,options);

fvs2 = [];
ngfs2 = [];
times2 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, nagents:%d\n', seed, K1, nagents);
    [x2, fv2, info2] = RFedAGSP(problem, x0, options);
    fvs2 = [fvs2, info2.fvs];
    ngfs2 = [ngfs2, info2.ngfs];
    times2 = [times2, info2.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs2, 2) - fv2_, '->', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times2,2), mean(fvs2,2) - fv2_,'->', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))


%% N = 100, S = 1000
options.option = 2;
nagents = 100; ncostterms = 1000;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
[x3_, fv3_, info3_] = rlbfgs(problem, x0,options);

fvs3 = [];
ngfs3 = [];
times3 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, nagents:%d\n', seed, K1, nagents);
    [x3, fv3, info3] = RFedAGSP(problem, x0, options);
    fvs3 = [fvs3, info3.fvs];
    ngfs3 = [ngfs3, info3.ngfs];
    times3 = [times3, info3.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs3,2) - fv3_, '-v', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times3,2), mean(fvs3,2) - fv3_,'-v', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% N = 150, S = 1000
nagents = 150; ncostterms = 1000;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
[x4_, fv4_, info4_] = rlbfgs(problem, x0,options);

fvs4 = [];
ngfs4 = [];
times4 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, nagents:%d\n', seed, K1, nagents);
    [x4, fv4, info4] = RFedAGSP(problem, x0, options);
    fvs4 = [fvs4, info4.fvs];
    ngfs4 = [ngfs4, info4.ngfs];
    times4 = [times4, info4.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs4,2) - fv4_, '-<', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

figure(2) 
hold on
semilogy(mean(times4,2), mean(fvs4,2) - fv4_,'-<', 'LineWidth',2,'MarkerIndices',1:gap:(options.maxiter / options.checkperiod))

%% N = 200, S = 500
nagents = 200; ncostterms = 1000;
[problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K1, pi);
[x5_, fv5_, info5_] = rlbfgs(problem, x0,options);

fvs5 = [];
ngfs5 = [];
times5 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, nagents:%d\n', seed, K1, nagents);
    [x5, fv5, info5] = RFedAGSP(problem, x0, options);
    fvs5 = [fvs5, info5.fvs];
    ngfs5 = [ngfs5, info5.ngfs];
    times5 = [times5, info5.CPUtime];
end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t,mean(fvs5,2)-fv5_, '-o', 'LineWidth', 2, 'MarkerIndices', 1:gap:(options.maxiter / options.checkperiod))
title("$S = 1000$", 'Interpreter','latex','FontSize',18)
ylabel('Optimality gap', 'FontSize',18,'Interpreter','latex')
xlabel('Iterations','FontSize',18,'Interpreter','latex')
legend('N=60', 'N=80', 'N=100', 'N=150', 'N=200', 'FontSize', 18)

saveas(gcf, filename + "_ER.fig")

figure(2) 
hold on
semilogy(mean(times5,2),mean(fvs5,2)-fv5_,'-o', 'LineWidth', 2, 'MarkerIndices', 1:gap:(options.maxiter / options.checkperiod))
ylabel('Optimality gap', 'FontSize',18,'Interpreter','latex')
xlabel('CPU time (s)','FontSize',18,'Interpreter','latex')
title("$S = 1000$", 'Interpreter','latex','FontSize',18)
legend('N=60', 'N=80', 'N=100', 'N=150', 'N=200', 'FontSize', 18)

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



