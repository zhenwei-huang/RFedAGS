set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

clear;close all;clc
rng(0);

load('RI.mat')

numSample = 20000;

train = RI(randperm(length(RI), numSample));

d = 9;
Manifold = sympositivedefinitefactory(d);
Manifold.type = 'SPD';
Manifold.retr = @Manifold.exp;
Manifold.invretr = @Manifold.log;
Manifold.transp = @Manifold.paralleltransp;


nagents = 50;
ncostterms = ceil(numSample/nagents);

prob = cell(nagents,1);
opts = cell(nagents,1);

K = 5;
pi = rand(nagents, 1);
problem.M = Manifold;
for i = 1:nagents
    Z = [];
    ttt = train((i-1)*ncostterms+1 : i*ncostterms);

    for j = 1:ncostterms
        D{j} = sqrtm(ttt{j});        
    end

    probtmp.M = Manifold;
    probtmp.ncostterms = ncostterms;
    probtmp = ConstructRFLFMCProb_SPD(D, ncostterms, probtmp);
    prob{i} = probtmp;

    optstmp.batchsize = ceil(0.5*ncostterms);
    optstmp.maxiter = K;       
    optstmp.pi = pi(i);
    opts{i} = optstmp;  

    % checkgradient(probtmp)
end

problem.nagents = nagents;
problem.agent = prob;
problem.cost = @(x) cost(problem, x);
problem.grad = @(x) grad(problem, x);
options.agent = opts;

% checkgradient(problem)

options.verbosity = 2;

options.checkperiod = 10;
options.maxiter = 300;

x0 = Manifold.rand();

[x_, fv_, info_] = rlbfgs(problem, x0);


%% RFedAGSP approximating 
options.option = 2;
options.s = 5;

Seed = 5;

fvs1 = [];
ngfs1 = [];
times1 = [];
options.stepsize = 1e-2;
options.localMethod = 'RSGD';
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedAGS\n', seed, K)
    % profile on
    [x1, fv1, info1] = RFedAGSP(problem, x0, options);
    infos1{seed} = info1;
    % profile viewer
    fvs1 = [fvs1, info1.fvs];
    times1 = [times1, info1.CPUtime];
end
figure(1)
t = 0:options.checkperiod:options.maxiter;
gap = ceil(length(t) / 40);
semilogy(t, mean(fvs1,2) - fv_, '-^', 'LineWidth',2, 'MarkerIndices',1:gap:length(t))

figure(2)
semilogy(mean(times1,2), mean(fvs1,2) - fv_,'-^', 'LineWidth',2,'MarkerIndices',1:gap:length(t))


%% RFedAvg 

fvs2 = [];
ngfs2 = [];
times2 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedAvg\n', seed, K)
    % profile on
    [x2, fv2, info2] = RFedAvg_(problem, x0, options);
    infos2{seed} = info2;
    % profile viewer
    fvs2 = [fvs2, info2.fvs];
    times2 = [times2, info2.CPUtime];

end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs2,2) - fv_, '->', 'LineWidth',2,'MarkerIndices',1:gap:length(t))


figure(2)
hold on
semilogy(mean(times2,2), mean(fvs2,2)-fv_,'->', 'LineWidth',2,'MarkerIndices',1:gap:length(t))


%% RFedSVRG

fvs3 = [];
ngfs3 = [];
times3 = [];
for seed = 1:Seed
    fprintf('seed:%d, K:%d, RFedSVRG\n', seed, K)
    % profile on
    [x3, fv3, info3] = RFedSVRG_(problem, x0, options);
    infos3{seed} = info3;
    % profile viewer
    fvs3 = [fvs3, info3.fvs];
    times3 = [times3, info3.CPUtime];

end
figure(1)
hold on
t = 0:options.checkperiod:options.maxiter;
semilogy(t, mean(fvs3,2)-fv_, '-v', 'LineWidth',2,'MarkerIndices',1:gap:length(t))
ylabel('Optimality gap','Interpreter','latex')
xlabel('Iterations','FontSize',18,'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'FontSize', 18)

figure(2)
hold on
semilogy(mean(times3,2), mean(fvs3,2)-fv_,'-v', 'LineWidth',2,'MarkerIndices',1:gap:length(t))
ylabel('Optimality gap','Interpreter','latex')
xlabel('CPU time','FontSize',18,'Interpreter','latex')
legend('RFedAGS', 'RFedAvg', 'RFedSVRG', 'FontSize', 18)



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


