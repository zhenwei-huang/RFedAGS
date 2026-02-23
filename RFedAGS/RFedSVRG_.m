function [x2, fv, info] = RFedSVRG_(problem, x, options)
    
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
    localgf = cell(problem.nagents, 1);
    
    CPUtime = 0; times = 0;cc=1;
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

        % compute full global gradient       
          % agents do
        time = toc(t1);
        local_times = zeros(length(St), 1);
        for i = 1:length(St)
            tic;
            prob = problem.agent{St(i)};            
            localgf{St(i)} = prob.grad(x1);
            local_times(i) = toc;
        end
        time = time + max(local_times);        

          % the server does
        t2 = tic;
        gfx1 = localgf{St(i)};
        for i = 2:length(St)
            gfx1 = gfx1 + localgf{St(i)};
        end
        gfx1 = gfx1 / problem.nagents;
        time = time + toc(t2);
           
        
        % active agents locally update        
        for i = reshape(St, 1, length(St)) % must be a row array
            % single agent 
            t3 = tic;
            prob = problem.agent{i};
            opts = options.agent{i};
            opts.stepsize_type = 'fix';
            opts.stepsize_init = options.stepsize;
            opts.option = options.option;              

            localx{i} = localSVRG(prob, x1, gfx1, opts);
            local_times(i) = toc(t3);
        end
        time = time + max(local_times);

        % the server aggratates
        t4 = tic;
        x2 = Aggregation(x1, localx, St, prob.M);
        
        t = t+1;
        x1 = x2;
        time = time + toc(t4);
        times = times + time;        

        if mod(t, options.checkperiod) == 0            
            xtmp{cc} = x2;  cc=cc+1;
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

function x = localSVRG(prob, x, gf, opts)
    maxiter = opts.maxiter;    
    stepsize = opts.stepsize_init;        
    x0 = x;
    iter = 0;
    
    while iter < maxiter        
        % SVRG step
        idx_batch = randperm(prob.ncostterms, opts.batchsize);                      
        eta1 = prob.partialgrad(x, idx_batch);
        eta2 = prob.partialgrad(x0, idx_batch);
        eta = eta1 - prob.M.transp(x0, x, eta2 - gf);

        % update local iterate
        x = prob.M.retr(x, eta, -stepsize);        
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
%         tmp = manifold.lincomb(x, 1, tmp, 1, manifold.invretr(x, localx{i}));
    end
    output = manifold.retr(x, tmp, 1/Nsub);
end

