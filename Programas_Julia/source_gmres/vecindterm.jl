function vecindterm(dim::Integer)
    v = zeros(dim);
    v[1] = 2.5;
    v[dim] = 2.5;
    n = floor(dim/2);
    for i = 2:dim-1
        if i == n || i == n+1
            v[i] = 1.0;
        else
            v[i] = 1.5;
        end
    end
    return v;
end