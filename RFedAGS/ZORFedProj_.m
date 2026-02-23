function [x_out, fv, info] = ZORFedProj_(problem, x, options)
    
    stepsize = options.stepsize;
    fv  = problem.cost(x);
    gf  = problem.grad(x);
    ngf = problem.M.norm(x, gf);    
    fvs  = fv;
    ngfs = ngf;
    x_old = x; x_new = x;
    stepsize_g = 1.;

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


    zeta_i = cell(problem.nagents, 1);
    z_hat_itK = cell(problem.nagents, 1); 
    x_mid = x_new;

    CPUtime = 0; times = 0;cc = 1;
    St_ = [];
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
        local_times = zeros(length(St), 1);
        for i = reshape(St, 1, length(St)) % must be a row array
            t2 = tic;
            % single agent 
            prob = problem.agent{i};
            opts = options.agent{i};            
            opts.stepsize_init = options.stepsize;
            opts.option = options.option;  
             
            if t == 0
                c_it = zeros(size(x_new));
                [z_hat_itK{i}, zeta_i{i}] = localUpdate(prob, x_new, c_it, opts);
            else
                if ismember(i, St_)
                    c_it = ((Proj(x_old) - x_mid) / (opts.maxiter * opts.stepsize_init * stepsize_g)) - zeta_i{i};
                else
                    c_it = zeros(size(x));
                end
                [z_hat_itK{i}, zeta_i{i}] = localUpdate(prob, x_mid, c_it, opts);                
            end
            local_times(i) = toc(t2);
        end
        time = time + max(local_times);

        % the server aggratates        
        t3 = tic;
        z_sum = z_hat_itK{St(1)}; 
        Nsub = length(St);
        for j = 2:Nsub
            z_sum = z_sum + z_hat_itK{St(j)};
        end
        x_new = (1 - stepsize_g) * Proj(x_mid) + (stepsize_g / Nsub) * z_sum;
        x_old = x_mid;
        x_mid = x_new; 
        t = t+1;
        x_tmp = Proj(x_new);  % ensure that each output is on the manifold.
        time = time + toc(t3);
        times = times + time;                
        
        if mod(t, options.checkperiod) == 0
            % t4 = tic;            
            % times = times + toc(t4);
            xtmp{cc} = x_tmp; cc = cc+1;
            fv   = problem.cost(x_tmp);
            gf   = problem.grad(x_tmp);        
            ngf  = problem.M.norm(x_tmp, gf);
            fvs  = [fvs;fv];
            ngfs = [ngfs;ngf];
            CPUtime =[CPUtime; CPUtime(end) + times];
            times = 0;
            if options.verbosity > 0
            fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|ss:%.4e|ans:%4d\n', t, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize, length(St));     
            % fprintf('iter:%6d|fv:%.6e(%.6e)|ngf:%.6e|ngf/ngf0:%.6e|stepsize:%.6e\n', t, fvs(end),fvs(end)/fvs(1),ngfs(end),ngfs(end)/ngfs(1),options.stepsize);                
            end
        end
        St_ = St;
    end 

    info.fvs  = fvs;    
    info.ngfs = ngfs;
    info.xs   = xtmp;
    info.CPUtime = CPUtime;
    x_out = x_tmp;    
end

function [out1, out2] = localUpdate(prob, x, c, opts)
    maxiter = opts.maxiter;    
    stepsize = opts.stepsize_init;
    z_hat_itk = Proj(x); z_itk = Proj(x);
    iter = 0;
    zeta = zeros(size(x));
    while iter < maxiter        
        idx_batch = randperm(prob.ncostterms, opts.batchsize);        
        % eta = prob.partialgrad(z_itk, idx_batch);
        eta = grad_estimator(z_itk, prob, idx_batch, opts.mu);
        z_hat_itk = z_hat_itk - stepsize * (eta + c);
        z_itk = Proj(z_hat_itk);
        zeta = zeta + eta;
        iter = iter + 1;
    end
    out1 = z_hat_itk;
    out2 = zeta / maxiter;
end

function out = Proj(X)
    [U,~,V] = svd(X,0);
    out = U*V';
end

function out = grad_estimator(x, prob, S, mu)
    m = length(S);
    [p, r] = size(x);
    out = zeros(p, r);
    for i = 1:m
        u = randn(p, r);
        u = u / norm(u, 'fro');
        tmp = Proj(x + mu * u);
        out = out + ((prob.partialcost(tmp, S(i)) - prob.partialcost(x, S(i))) / mu) * u;
    end
    out = out * ( p * r / m);
end

