using LinearAlgebra

function sor_method(D, L, U, b, x0, ω, iter_max, tol)
    dim = length(b)
    xp = zeros(dim)
    xn = zeros(dim)
    iter = 0
    error_method = 0.0
    xp = x0
    xe = ones(dim)

    T = D+ω*L
    invT = inv(T)

    ωD = (1-ω)*D
    ωU = ω*U
    f = ω*invT*b

    for k = 1:iter_max
        xn = invT*(ωD*xp - ωU*xp) + f
        error_method = norm(xn - xe) / norm(xn)
        if error_method <= tol
            iter = k
            break
        end
        xp = xn
    end

    return error_method, iter

end