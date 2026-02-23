function [fvs, times, numgfs, rmses, problem] = LRMCGrass_testing_ML_fixed(x0, K, option, r, lambda, Seed, stepsize, pi)

    rng(1)

    MLdata1M;

    num_cols = nusers;
    % num_cols = 2000;
    m = nmovies;                    
       
    problem.M = grassmannfactory(m, r);
    problem.M.retr = @retr_cayley;
    problem.M.invretr = @invretr_cayley;
    problem.M.transp = @vt_cayley;
    
    nagents = 40;
    ncostterms = num_cols / nagents;

    prob = cell(nagents,1);
    opts = cell(nagents,1);    
    for i = 1:nagents
        probtmp.M = problem.M;
        probtmp.ncostterms = ncostterms;       
        probtmp = ConstructRFLLRMCProb_Grass_({dataset_train{(i-1)*ncostterms + 1 : i * ncostterms}}, ...
                                              ncostterms, lambda, probtmp);
        prob{i} = probtmp;
    
        optstmp.batchsize = ceil(ncostterms * 0.1);
        optstmp.maxiter = K;       
        optstmp.pi = pi(i);
        opts{i} = optstmp;
        % checkgradient(probtmp)
    end
    
    problem.nagents = nagents;
    problem.agent = prob;
    problem.cost = @(x) cost(problem, x);
    problem.grad = @(x) grad(problem, x);
    problem.completion_Matrix = @(U, D) completion_Matrix(U, D, problem);
    options.agent = opts;
    % checkgradient(problem)      
    
    options.checkperiod = 1;
    options.maxiter = 40;
    options.option = option;  % estimating
    options.stepsize = stepsize;
    options.localMethod = 'RSGD';
    options.stepsize_type = 'fix';
    
    fvs = [];
    ngfs = [];
    numgfs = [];
    times = [];
    rmses = [];
    for seed = 1:Seed
        fprintf('seed:%d, K:%d\n', seed, K)
        [x, fv, info] = RFedAGSP(problem, x0, options);
        fvs = [fvs, info.fvs];
        ngfs = [ngfs, info.ngfs];
        numgfs = [numgfs, info.numgfs];
        times = [times, info.CPUtime];  

        rmse = probtmp.RMSE(x0, {dataset_test{1:num_cols}});
        for i = 1:options.maxiter
            rmse(end+1) = probtmp.RMSE(info.xs{i}, {dataset_test{1:num_cols}});
        end
        rmses = [rmses, reshape(rmse, length(rmse), 1)];  
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

function out = retr_cayley(x, u, t)    
    if nargin == 3 && abs(t - 1) > eps
        u = t * u;
    end
    tmp = u' * u;    
    out = x + u - ((0.5 * x + 0.25 * u) / (eye(size(x,2)) + 0.25 * tmp)) * tmp;
end

function out = invretr_cayley(x, y)
    tmp = x' * y;
    out = 2 * (y - x * tmp)/(eye(size(x, 2)) + tmp);
end

function out = vt_cayley(x, y, u)
    v = invretr_cayley(x, y);
    out = u - ((x + 0.5 * v) / (eye(size(x, 2)) + 0.25 * (v' * v))) * (v' * u);
end


