function fns = ConstructRFLPCAProb_St_(D, N, fns)
% % D is the dataset, stroed in cell.
% % N is number of sample
% % fns is a structure related to Problem, and contains 
% %    fns.cost,
% %    fns.grad,
% %    fns.partialgrad, 
% 
%     fns.cost  = @(x) cost(x, D, N);
%     fns.grad = @(x) partialgrad(x, D, 1:N);
%     fns.partialgrad = @(x, S) partialgrad(x, D, S);    
% end
% 
% function output = cost(x, D, N)
%     Dx = D*x;
%     output = - (Dx'*Dx) / N;
% end
% 
% function output = partialgrad(x, D, S)
%     S = sort(S);
%     Ds = D(S,:);
%     tmp = (-2 / length(S)) * Ds' * (Ds * x);   
%     output = tmp - (x'*tmp) * x;
%     % output = tmp;
%     % / length(S);
% end


    fns.cost  = @(x) cost(x, D, N);
    fns.partialcost = @(x, S) partialcost(x, D, S);
    fns.egrad = @(x) partialegrad(x, D, 1:N);
    fns.partialegrad = @(x, S) partialegrad(x, D, S);
    
    fns.grad = @(x) fns.M.egrad2rgrad(x, fns.egrad(x));
    fns.partialgrad = @(x, S) fns.M.egrad2rgrad(x, fns.partialegrad(x, S));

end

function output = cost(x, D, N)
    tmp = 0;
    for i = 1:N
        tmp = tmp + singlecost(x, D{i});
    end
    output = tmp / N;
end

function output = partialcost(x, D, S)
    output = 0;
    for i = reshape(S, 1, length(S))
        output = output + singlecost(x, D{i});
    end
    output = output / length(S);
end

function output = partialegrad(x, D, S)
    tmp = zeros(size(x));
    for i = reshape(S, 1, length(S))
        tmp = tmp + singleegrad(x, D{i});
    end
    output = tmp / length(S);
end

function output = singlecost(x, Di)
    tmp = Di' * x;
    output = - trace(tmp' * tmp);
end

function output = singleegrad(x, Di)    
    output = (-2) * Di * (Di' * x);
end


