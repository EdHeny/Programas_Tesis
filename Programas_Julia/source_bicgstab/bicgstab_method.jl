using LinearAlgebra

function bicgstab_method(A, b, x0, tol)
    dim = size(A, 1)
    iter_max = dim

    xe = ones(dim) # solution system
    error_method = 0.0
    iter = 0
    k = 0

    xp = zeros(dim)
    rp = zeros(dim)
    dp = zeros(dim)
    rast = zeros(dim)

    xn = zeros(dim)
    rn = zeros(dim)
    dn = zeros(dim)

    xp = x0
    rp = b - A*xp
    dp = rp
    rast = rp

    for k in 1:iter_max
        Ad = A*dp
        α = dot(rp, rast) / dot(Ad, rast)
        s = rp - α*Ad
        As = A*s
        ω = dot(As, s) / dot(As, As)
        xn = xp + α*dp + ω*s
        error_method = norm(xn - xe)/norm(xn)
        if error_method <= tol
            iter = k
            break
        end
        rn = s - ω*As
        β = (dot(rn, rast)/dot(rp, rast)) * (α/ω)
        dn = rn + β*(dp-ω*Ad)
        
        # Actualizar
        xp = xn
        rp = rn
        dp = dn
    end

    return error_method, iter, xn
end