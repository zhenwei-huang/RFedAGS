function output = lor2poin(emLor)
    [d, n] = size(emLor);
    tmp = zeros(d-1, n);
    for i = 1:n
        tmp(:,i) = emLor(2:end, i) / (emLor(1, i) + 1);
    end
    output = tmp;
end

