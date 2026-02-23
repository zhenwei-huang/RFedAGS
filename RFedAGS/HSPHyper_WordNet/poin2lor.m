function output = poin2lor(emPoin)
    [d, n] = size(emPoin);
    tmp = zeros(d+1, n);    
    for i = 1:n
        sqnorm = emPoin(:, i)' * emPoin(:, i);
        tmp(1,i)   = (1 + sqnorm) / (1 - sqnorm);
        tmp(2:end, i) = 2 * emPoin(:,i) / (1 - sqnorm);
    end
    output = tmp;
end

