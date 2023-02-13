using LinearAlgebra

function cg_method(A, b, x0, tol)
    dim = size(A, 1)
    iter_max = dim

    xe = ones(dim) # solution system
    error_method = 0.0
    iter = 0
    k = 0

    xp = zeros(dim)
    rp = zeros(dim)
    dp = zeros(dim)

    xn = zeros(dim)
    rn = zeros(dim)
    dn = zeros(dim)

    xp = x0
    rp = b - A*xp
    dp = rp

    for k in 1:iter_max
        Adp = A*dp
        α = dot(rp, rp) / dot(Adp, dp)
        xn = xp + α*dp
        # Convergece Criteria
        error_method = norm(xn - xe)/norm(xn)
        if error_method <= tol
            iter = k
            break
        end
        rn = rp - α*Adp
        β = dot(rn, rn)/dot(rp, rp)
        dn = rn + β*dp
        
        # Actualizar
        xp = xn
        rp = rn
        dp = dn
    end

    return error_method, iter, xn
end