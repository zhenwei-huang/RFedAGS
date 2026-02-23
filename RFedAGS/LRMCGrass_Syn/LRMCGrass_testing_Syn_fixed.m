function [fvs, times, numgfs, rel_errs, problem] = LRMCGrass_testing_Syn_fixed(x0, K, option, s, Seed, stepsize, pi)

    rng(1)

    num_cols = 2000;
    m = 200;
    r = 5; 
    
    A = randn(m, r); 
    B = randn(num_cols, r);
    dim = m * r + num_cols * r - r*r;
    
    AB = A * B';
    nAB = norm(AB, 'fro');
    
    OS = 6;
    
    Num_Omega = OS * dim;
    
    remove_p = randperm(num_cols * m, num_cols * m - Num_Omega);
    
    reshaped_AB = AB(:) + 1e-6 * randn(num_cols * m, 1);
    reshaped_AB(remove_p) = 0;
    
    removed_AB = reshape(reshaped_AB, m, num_cols);
    
    Omega = cell(num_cols, 1);
    for i = 1:num_cols
        Omega{i} = find(removed_AB(:,i)~=0);
    end
    
    lambda = 0;
    problem.M = grassmannfactory(m, r);
    problem.M.retr = @retr_cayley;
    problem.M.invretr = @invretr_cayley;
    problem.M.transp = @vt_cayley;
    
    nagents = 20;
    ncostterms = num_cols / nagents;

    prob = cell(nagents,1);
    opts = cell(nagents,1);
    D = cell(nagents, 1);
    for i = 1:nagents
        D{i} = cell(1, ncostterms);
        for j = 1:ncostterms
            D{i}{j}.X = removed_AB(:, (i-1)*ncostterms+j);
            D{i}{j}.Omega = Omega{(i-1)*ncostterms+j};
        end
        probtmp.M = problem.M;
        probtmp.ncostterms = ncostterms;
        probtmp = ConstructRFLLRMCProb_Grass_(D{i}, ncostterms, lambda, probtmp);
        prob{i} = probtmp;
    
        optstmp.batchsize = ceil(ncostterms * 0.5);
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
    options.maxiter = 100;
    options.option = option;  % estimating
    options.stepsize = stepsize;
    options.localMethod = 'RSGD';
    options.stepsize_type = 'fix';
    
    fvs = [];
    ngfs = [];
    numgfs = [];
    times = [];
    rel_errs = [];
    for seed = 1:Seed
        fprintf('seed:%d, K:%d\n', seed, K)
        [x, fv, info] = RFedAGSP(problem, x0, options);
        fvs = [fvs, info.fvs];
        ngfs = [ngfs, info.ngfs];
        numgfs = [numgfs, info.numgfs];
        times = [times, info.CPUtime];
        rel_err = norm(problem.completion_Matrix(x0, D) - AB, 'fro') / nAB;
        for i = 1:length(info.xs)
            rel_err(end+1) = norm(problem.completion_Matrix(info.xs{i}, D) - AB, 'fro') / nAB;
        end
        rel_errs = [rel_errs, reshape(rel_err, length(rel_err), 1)];
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


function out = completion_Matrix(U, D, problem)
    out = [];
    for i = 1:problem.nagents
        out = [out, problem.agent{i}.completion(U, D{i})];
    end
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


