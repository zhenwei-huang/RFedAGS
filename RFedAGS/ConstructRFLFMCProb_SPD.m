function fns = ConstructRFLFMCProb_SPD(D, N, fns)
% D is the dataset, stroed in cell.
% N is number of sample
% fns is a structure related to Problem, and contains 
%    fns.cost, 
%    fns.egrad,
%    fns.partialegrad, 

    fns.cost  = @(x) cost(x, D);    
    fns.grad = @(x) partialgrad(x, D, 1:N);
    fns.partialgrad = @(x, S) partialgrad(x, D, S);

    
    trATB = @(A, B) sum(sum(A.*B));   % trace(A'*B);
    trA2 = @(A) trATB(A, A);          % trace(A^T*A), \|A\|_F^2
    
    function output = cost(x, D)
        tmp = 0;
        for i = 1:N
            tmp = tmp + singlecost(x, D{i});
        end
        output = tmp / N;
    end    
    
    % partial Riemannian gradient
    function output = partialgrad(x, D, S)
        tmp = zeros(size(x));
        for i = reshape(S, 1, length(S))
            tmp = tmp + singlegrad(x, D{i});
        end
        output = tmp / length(S);
    end
    
    function output = singlecost(x, Di)
        sm = sqrtm(x);
        tmp = logm(sm \ Di / sm);
        output = 0.5 * trA2(real(tmp));
    end
    

    % Riemannian gradient for f_ij
    function output = singlegrad(x,  Di)
        output = - fns.M.log(x, Di);
    end

end