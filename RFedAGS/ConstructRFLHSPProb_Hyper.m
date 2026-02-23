function fns = ConstructRFLHSPProb_Hyper(Y, alpha, N, fns)
% W is the features of class samples, stored in cell;
% Y is the hyperbolic embeddings of the class of W, stroed in cell.
% N is number of sample
% fns is a structure related to Problem, and contains 
%    fns.cost, 
%    fns.egrad,
%    fns.partialegrad, 
        
    fns.cost  = @(x) cost(x);          
    fns.grad = @(x) partialgrad(x, 1:N, alpha, Y);
    fns.partialgrad = @(x, S) partialgrad(x, S, alpha, Y);

    function output = cost(x)
        tmp = singlecost(x, alpha(1), Y{1});
        for i = 2:N
            tmp = tmp + singlecost(x, alpha(i), Y{i});
        end
        output = tmp / N;
    end    
    
    % partial Riemannian gradient
    function output = partialgrad(x, S, alpha, Y)
        lS = length(S);
        tmp = singlegrad(x, alpha(S(1)), Y{S(1)});        
        for i = 2:lS
            tmp = tmp + singlegrad(x, alpha(S(i)), Y{S(i)});
        end
        output = tmp / lS;
    end
    
    % cost for f_ij
    function output = singlecost(x, alphai, Yi)
        d = fns.M.dist(x, Yi);
        output = alphai * d * d;
    end    
        
    % Riemannian gradient for f_ij
    function output = singlegrad(x, alphai, Yi)
        output = - 2 * alphai * fns.M.log(x, Yi);
    end

end