function [x2, fv, info] = RFedAvg_(problem, x, options)
    
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
    
    if options.verbosity > 0
%     fprintf('iter:%6d,fv:%.6e(%.6e),ngf:%.6e,ngf/ngf0:%.6e,stepsize:%.6e\n', 0, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize);
    fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|ss:%.6e\n', 0, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize);
    end
    t = 0; ct = 0;   
    if options.checkperiod > 0
        xtmp = cell(options.maxiter / options.checkperiod, 1);    
    else
        xtmp = cell(options.maxiter, 1);
    end
            
    localx  = cell(problem.nagents, 1);
    
    CPUtime = 0; times = 0; cc=1;
    while t < options.maxiter        
        t1 = tic;
        if (strcmp(options.stepsize_type, 'decay')) && (mod(t, options.decaysteps) == 0) && (t>0)                                        
            ct = ct + 1;
            options.stepsize = stepsize / (options.lambda + ct);
        end
  
        % server sampling agents
        if options.option == 1     % full agents
            St = 1:problem.nagents;   
        elseif options.option == 2 % probability
            St = [];
            for i = 1:problem.nagents
                opts = options.agent{i};
                pi = opts.pi;
                if rand(1) <= pi
                    St(end+1) = i;
                end
            end
            if isempty(St)
                fprintf('empty!\n')
                St = randi(problem.nagents);          
            end
        elseif options.option == 3 % sampling without replacement
            St = sort(randperm(problem.nagents, options.s));
        end 

        % active agents locally update
        time = toc(t1);
        local_time = zeros(length(St),1);
        for i = reshape(St, 1, length(St)) % must be a row array
            t2 = tic;
            % single agent 
            prob = problem.agent{i};
            opts = options.agent{i};
            opts.stepsize_type = 'fix';
            opts.stepsize_init = options.stepsize;
            opts.option = options.option;  
            
            localx{i} = localRSGD(prob, x1, opts);

            local_time(i) = toc(t2);
        end
        time = time + max(local_time);

        % the server aggratates
        t3 = tic;
        x2 = Aggregation(x1, localx, St, prob.M);        
        t = t+1;
        x1 = x2;
        time = time + toc(t3);
        times = times + time;        
        
        if mod(t, options.checkperiod) == 0            
            xtmp{cc} = x2; cc = cc+1;
            fv   = problem.cost(x2);
            gf   = problem.grad(x2);        
            ngf  = problem.M.norm(x2, gf);
            fvs  = [fvs;fv];
            ngfs = [ngfs;ngf];
            CPUtime =[CPUtime; CPUtime(end) + times];
            times = 0;
            if options.verbosity > 0
            fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|ss:%.6e|ans:%6d\n', t, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize, length(St));     
            % fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|stepsize:%.6e\n', t, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize);        
            end
        end                 
    end 

    info.fvs  = fvs;    
    info.ngfs = ngfs;
    info.xs   = xtmp;
    info.CPUtime = CPUtime;
end

function x = localRSGD(prob, x, opts)
    maxiter = opts.maxiter;    
    stepsize = opts.stepsize_init;    
    iter = 0;
    while iter < maxiter        
        idx_batch = randperm(prob.ncostterms, opts.batchsize);
        eta = - stepsize * prob.partialgrad(x, idx_batch);

        % update local iterate
        x = prob.M.retr(x, eta, 1);        
        iter = iter + 1;
    end
end

function output = Aggregation(x, localx, St, manifold)    
    Nsub = length(St);
    tmp = zeros(size(x));
    if isfield(manifold, 'log')
        manifold.invretr = @manifold.log;
    end
    if isfield(manifold, 'exp')
        manifold.retr = @manifold.exp;
    end    

    for i = 1:Nsub
        tmp = tmp + manifold.invretr(x, localx{St(i)});
    end
    output = manifold.retr(x, tmp, 1/Nsub);
end

