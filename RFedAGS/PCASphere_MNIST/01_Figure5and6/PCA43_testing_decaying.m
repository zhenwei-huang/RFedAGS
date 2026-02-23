function [fvs, times, numgfs, problem, fvs_, problem_] = PCA41_testing_fixed(iidtype, x0, K, option, s, Seed, stepsize, pi)
    d = 784;
    Manifold = spherefactory(d, 1);
    Manifold.retr = @Manifold.exp;
    Manifold.invretr = @Manifold.log;
    Manifold.transp = @Manifold.isotransp;
    problem.M = Manifold;

    if iidtype == 0
        load('MNIST_train_60_iid.mat')
    elseif iidtype == 1
        load('MNIST_train_60_noniid_small.mat')
    elseif iidtype == 2
        load('MNIST_train_60_noniid_large.mat')
    end
    nagents = double(nagents);
    ncostterms = double(ncostterms);

    options.option = option; 
    options.s = s;
    options.localMethod = 'RSGD';
    options.stepsize = stepsize;
    options.checkperiod = 10;
    options.maxiter = 1000;
    options.stepsize_type = 'decay';
    options.decaysteps = 20;
    [problem, problem_, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K, pi);

    fvs = []; fvs_ = [];   
    numgfs = [];
    times = [];    
    for seed = 1:Seed
        fprintf('seed:%d, K:%d\n', seed, K)
        % profile on
        [x, fv, info] = RFedAGSP__(problem, x0, options);
        % profile viewer
        fvs = [fvs, info.fvs];
        numgfs = [numgfs, info.numgfs];
        times = [times, info.CPUtime];
        
        fv_ = problem_.cost(x0);        
        array = 0:options.checkperiod:length(info.xs);
        for i = array(2:end)            
            fv_(end+1) = problem_.cost(info.xs{i});            
        end
        fvs_ = [fvs_, reshape(fv_, length(fv_), 1)];
    end
end


function [problem1, problem2, options] = construct_prob(D, problem1, options, nagents, ncostterms, batchsize, K, pi)
    
    prob1 = cell(nagents,1);
    prob2 = cell(nagents,1);
    opts = cell(nagents,1);
    for i = 1:nagents
        Z = cell(ncostterms,1);
        for j = 1:ncostterms
            Z{j} = D((i-1)*ncostterms + j, :)';
        end 
        probtmp.M = problem1.M;
        probtmp.ncostterms = ncostterms;
        probtmp = ConstructRFLPCAProb_St_(Z, ncostterms, probtmp);
        prob1{i} = probtmp;

        f = @(x) prodij(x, i, pi);
        pi_til = pi(i) * integral(f, 0, 1);
        prob2{i}.cost = @(x) pi_til * probtmp.cost(x);
        prob2{i}.partialgrad = @(x) pi_til * probtmp.partialgrad(x);
        prob2{i}.grad = @(x) pi_til * probtmp.grad(x);
        
        optstmp.batchsize = batchsize;
        optstmp.maxiter = K;       
        optstmp.pi = pi(i);
        opts{i} = optstmp;                 
    end
    problem1.nagents = nagents;
    problem1.agent = prob1;
    problem1.cost = @(x) cost(problem1, x);
    problem1.grad = @(x) grad(problem1, x);
    options.agent = opts;

    problem2.M = problem1.M;
    problem2.nagents = nagents;
    problem2.agent = prob2;
    problem2.cost = @(x) cost(problem2, x);
    problem2.grad = @(x) grad(problem2, x);
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


function out = prodij(t,i,pi)
    n = length(pi);
    out = 1;
    for j = 1:n
        if j ~= i
            out = out .* (1-pi(j) + pi(j)*t);
        end
    end
end