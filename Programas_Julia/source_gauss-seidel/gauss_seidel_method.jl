using LinearAlgebra

function gauss_seidel_method(D, L, U, b, x0, iter_max, tol)
    dim = length(b)
    xp = zeros(dim)
    xn = zeros(dim)
    iter = 0
    error_method = 0.0
    xp = x0
    xe = ones(dim)

    T = D+L
    invT = inv(T)

    for k = 1:iter_max
        xn = invT*(b-U*xp)
        error_method = norm(xn - xe) / norm(xn)
        if error_method <= tol
            iter = k
            break
        end
        xp = xn
    end

    return error_method, iter

end