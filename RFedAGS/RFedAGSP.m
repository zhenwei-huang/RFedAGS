function [x2, fv, info] = RFedAGSP(problem, x, options)
    
    stepsize = options.stepsize;
    fv  = problem.cost(x);
    gf  = problem.grad(x);
    ngf = problem.M.norm(x, gf);    
    fvs  = fv;
    ngfs = ngf;
    x1 = x; x2 = x;

    if ~isfield(options, 'stepsize_type')
        options.stepsize_type = 'fix';
    elseif ~isfield(options, 'lambda')
        options.lambda = 1;
    end
    
    fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|ss:%.6e\n', 0, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize);
    
    t = 0; ct = 0;
    if options.checkperiod > 0
        xtmp = cell(options.maxiter / options.checkperiod, 1);    
    else
        xtmp = cell(options.maxiter, 1);
    end    
    numgfs = 0;
    localzeta  = cell(problem.nagents, 1);
    qit = zeros(problem.nagents,1);
    numgf_ = 0;
    CPUtime = 0; times = 0; cc = 1;
    while t < options.maxiter    
        t1 = tic;
        weights = zeros(problem.nagents, 1);        
        if (strcmp(options.stepsize_type, 'decay')) && (mod(t, options.decaysteps) == 0) && (t>0)                                        
            ct = ct + 1;
            options.stepsize = stepsize / (options.lambda + ct);
        end
        
        % server sampling agents
        if options.option == 1     % full agents
            St = 1:problem.nagents;   
            qit(St) = t+1;            
        elseif options.option == 2 % approximating probability
            St = [];
            for i = 1:problem.nagents
                opts = options.agent{i};
                pi = opts.pi;
                if rand(1) <= pi
                    St(end+1) = i;
                    qit(i) = qit(i) + 1;
                end
            end
            if isempty(St)
                fprintf('empty!\n')
                St = randi(problem.nagents);
                qit(St) = qit(St) + 1;                
            end
        elseif options.option == 3 % sampling without replacement
            St = sort(randperm(problem.nagents, options.s));
            qit(St) = t+1;
        elseif options.option == 4 % true probabilities 
            St = [];
            for i = 1:problem.nagents
                opts = options.agent{i};
                pi = opts.pi;
                if rand(1) <= pi
                    St(end+1) = i;
                    qit(i) = pi * (t+1);
                end
            end
        end        
        
        time = toc(t1);
        % active agents locally update        
        local_times = [];
        for i = reshape(St, 1, length(St)) % must be a row array
            % single agent
            t2 = tic;
            prob = problem.agent{i};
            opts = options.agent{i};
            opts.stepsize = options.stepsize;            
            
            weights(i) = prob.ncostterms;
            if     (strcmp(options.localMethod, 'RSGD'))                
                [~, localzeta{i}, numgf] = localRSGD(prob, x1, opts);
            elseif (strcmp(options.localMethod, 'RSVRG'))             
                [~, localzeta{i}, numgf] = localRSVRG(prob, x1, opts);
            else
                error('No local method "%s" exists!', options.localMethod);
            end
            local_times(end+1) = toc(t2);
        end
        time = time + max(local_times);
        % the server aggratates 
        t3 = tic;
        x2 = Aggregation(x1, localzeta, weights, qit / (t+1), St, problem.nagents, options.option, prob.M);
        numgf_ = numgf_ + numgf;
        t = t + 1;
        x1 = x2;
        time = time + toc(t3);
        times = times + time;

        if mod(t, options.checkperiod) == 0            
            xtmp{cc} = x2; cc = cc + 1;
            fv   = problem.cost(x2);
            gf   = problem.grad(x2);        
            ngf  = problem.M.norm(x2, gf);
            fvs  = [fvs;fv];
            ngfs = [ngfs;ngf];
            numgfs(end+1)  = numgfs(end) + numgf_;
            CPUtime(end+1) = CPUtime(end) + times;
            times = 0;
            numgf_ = 0;
            fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|ss:%.6e,ans:%4d\n', t, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize,length(St));        
        end                 
    end 

    info.fvs  = fvs;    
    info.ngfs = ngfs;
    info.xs   = xtmp;
    info.numgfs = reshape(numgfs, length(numgfs), 1);
    info.CPUtime = reshape(CPUtime, length(CPUtime), 1);
    info.qit = qit;
end

function [x1, zeta, numgf] = localRSGD(prob, x0, opts)
    maxiter = opts.maxiter;
    stepsize = opts.stepsize;    
    x1 = x0; x2 = x0;
    numgf = 0;    
    for iter = 1:maxiter        
        idx_batch = sort(randperm(prob.ncostterms, opts.batchsize)); numgf = numgf + opts.batchsize;
        eta = - stepsize * prob.partialgrad(x1, idx_batch);                     

        % update local iterate
        x2 = prob.M.retr(x1, eta);

        % gradient stream
        if iter == 1
            zeta = eta;
        else
            zeta = zeta + prob.M.transp(x1, x0, eta);
%             prob.M.lincomb(x0, 1, zeta, 1, prob.M.transp(x, x0, eta));
        end
        x1 = x2;
    end
end

function [x1, zeta, numgf] = localRSVRG(prob, x, opts)
    maxiter = opts.maxiter;    
    stepsize = opts.stepsize;
    x1 = x; x2 = x;
    numgf = 0;        
    g = prob.grad(x); numgf = numgf + prob.ncostterms;    
    for iter = 1:maxiter
        idx_batch = sort(randperm(prob.ncostterms, opts.batchsize)); 
        numgf = numgf + opts.batchsize * 2; 
        tmp1 = prob.partialgrad(x, idx_batch) - g;
        tmp2 = prob.partialgrad(x1, idx_batch);
        eta = - stepsize * (tmp2 - prob.M.transp(x, x1, tmp1));
        
        % update local iterate
        x2 = prob.M.retr(x1, eta);

        % gradient stream
        if iter == 1
            zeta = - stepsize * g;
        else
            zeta = zeta - stepsize * (prob.M.transp(x1, x, tmp2) - tmp1);
        end
        x1 = x2;
    end
end

function output = Aggregation(x, localzeta, weights, qit, St, N, option, manifold) 
% qit    : approximating probability
% St     : active agents at the t-th communication round
% N      : number of all agents
% option : sampling scheme

    if option ~= 2 && option ~= 4 
        sumWeights = sum(weights);
        Delta = weights(St(1)) * localzeta{St(1)};
        for i = reshape(St(2:end), 1, length(St(2:end)))
            Delta = Delta + weights(i) * localzeta{i};
        end 
        output = manifold.retr(x, Delta, 1 / sumWeights);
    else
        Delta = (1 / qit(St(1))) * localzeta{St(1)};
        for i = reshape(St(2:end), 1, length(St(2:end)))
            Delta = Delta + (1 / qit(i)) * localzeta{i};
        end 
        output = manifold.retr(x, Delta, 1 / N);
    end    
end

