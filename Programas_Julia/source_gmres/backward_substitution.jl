function backward_substitution(U, b, m)
    x = zeros(m);
    x[m] = b[m] / U[m,m];
    for i in m-1:-1:1
        s = dot(U[i,i+1:m],x[i+1:m]);
        x[i] = (b[i] - s) / U[i,i];
    end
    return x
end