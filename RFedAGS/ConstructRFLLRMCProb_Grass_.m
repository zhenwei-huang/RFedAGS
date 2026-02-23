function fns = ConstructRFLLRMCProb_Grass_(D, N, lambda, fns)
% D is the dataset, stroed in cell.
%   D{i}.X                     : the i-th column of D
%   D{i}.Omega = {j_1,j_2,...} : the indices of the known elements according to Di.X
% lambda                       : regularization parameter
% N is number of sample, i.e., N = |D|
% fns is a structure related to Problem, and contains 
%    fns.cost, 
%    fns.grad,
%    fns.partialgrad,    

    fns.cost  = @(U) cost(U, D, N, lambda);
    fns.grad = @(U) partialgrad(U, D, lambda, 1:N);
    fns.partialgrad = @(U, S) partialgrad(U, D, lambda, S);

    fns.completion = @(U, D_testing) completion(U, D_testing);
    fns.RMSE = @(U, D_testing) RMSE(U, D_testing);
         
    trATB = @(A, B) sum(sum(A.*B)); % trace(A^T*B)
    trA2  = @(A) (trATB(A, A));     % (trace(A^TA)) , \|A\|_F^2
    
    function output = cost(U, D, N, lambda)
        tmp = 0;
        for i = 1:N
            tmp = tmp + singlecost(U, D{i}, lambda);
        end
        output = tmp / N;
    end
    
    % partial Euclidean gradient
    function output = partialgrad(U, D, lambda, S)
        tmp = zeros(size(U));
        for i = reshape(S, 1, length(S))
            tmp = tmp + singlegrad(U, D{i}, lambda);
        end
        output = tmp / length(S);
    end

    function out = completion(U, D)
        [~, r] = size(U);
        n = length(D);
        W = zeros(n, r);
        for i = 1:n
            W(i,:) = compute_wU(U, D{i}, lambda);
        end
        out = U * W';
    end

    function out = RMSE(U, D)        
        % Xhat = completion(U, D);
        % tmp = 0;
        % m = 0;
        % for i = 1:N
        %     Xi = D{i}.X;
        %     Oi = D{i}.Omega;
        %     Xi_O = Xi(Oi);
        %     Xhat_i = Xhat(Oi,i);
        %     tmp = tmp + trA2(Xi_O - Xhat_i);
        %     m = m + length(Oi);
        % end
        % out = sqrt(tmp / m);
        nlen = length(D);
        tmp = 0;
        m = 0;
        for i = 1:nlen
            Xi = D{i}.X;
            Oi = D{i}.Omega;
            Xhat_i = completion(U, D(i));
            tmp = tmp + trA2(Xi(Oi) - Xhat_i(Oi));            
            m = m + length(Oi);
        end
        out = sqrt(tmp / m);
    end
    

%%%%%%%%%%%%%%%%%%%%% single point %%%%%%%%%%%%%%%%%
    % function value f_ij 
    function output = singlecost(U, Di, lambda)
        X = Di.X;
        Omega = Di.Omega;        
        wU = compute_wU(U, Di, lambda); 
        
        tmp0 = U * wU';
        tmp1 = PO(Omega, tmp0 - X);
        tmp2 = tmp0 - PO(Omega, tmp0);
        output = 0.5 * (trA2(tmp1) + lambda * trA2(tmp2));
    end
    
    % Riemannian gradient for f_ij
    function output = singlegrad(U, Di, lambda)
        [wU, rU] = compute_wU(U, Di, lambda);
        output = rU * wU + lambda * (U * wU') * wU;
    end


    function [wU, rU] = compute_wU(U, Di, lambda)
        X = Di.X;
        O = Di.Omega;
        r = size(U,2);
        [m,n] = size(X);
        C2 = PO(O, ones(m, n));
        Lambda = lambda * C2;
        Chat = C2 - Lambda;
        B = Chat(:);
%         InU = kron(eye(n), U);
%         A = (U' .* B') * U + lambda * eye(r*n);
        A = (U' .* B') * U;
        A(1:r+1:r*r) = A(1:r+1:r*r) + lambda * ones(1,r*n);
        X_O = PO(O, X);
        tmp = U' * (C2 .* X_O);
        wU = (full(A) \ tmp(:))';
%         wU = reshape(vecwU, n, r);       
        rU = Chat .* (U * wU' - X_O) - lambda * X_O;
    end

    function out = PO(Omega, X)        
        out = zeros(size(X));
        out(Omega) = X(Omega);
%         n = length(Omega);
%         for i = 1:n
%             ind = Omega(i);
%             out(ind) = X(ind);
%         end
    end    
end