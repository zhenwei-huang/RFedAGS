function [fvs, times, numgfs, problem] = PCA42_testing_fixed(iidtype, x0, K, option, s, Seed, stepsize, pi)
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
    options.maxiter = 500;
    options.stepsize_type = 'fix';
    [problem, options] = construct_prob(data, problem, options, nagents, ncostterms, ceil(ncostterms * 0.5), K, pi);

    fvs = [];   
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
    end
end


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
