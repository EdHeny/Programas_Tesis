using LinearAlgebra
function coefmatsys(dim::Integer)
    d = 3*ones(dim);
    du = -1*ones(dim-1);
    dl = -1*ones(dim-1);
    M = Tridiagonal(dl, d, du);
    M = M + zeros(dim, dim);
    for i = 1:dim
        if M[i,dim-i+1] == 0;
            M[i,dim-i+1] = 0.5;
        end
    end
    return M;
end