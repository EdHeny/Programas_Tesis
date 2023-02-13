using BenchmarkTools

include("coefmatsys.jl")
include("vecindterm.jl")
include("sor_method.jl")

function main(dim)
    A = coefmatsys(dim);
    b = vecindterm(dim);
    x0 = zeros(dim);
    ω = 1.13
    iter_max = 60000
    tol = 1.0e-10;

    # Descomposición A = D + L + U
    D =  Diagonal(A)
    L = tril(A,-1)
    U = triu(A,1)

    println(raw"Successive Over-Relaxation Method")
    info_method = sor_method(D, L, U, b, x0, ω, iter_max, tol);
    @time sor_method(D, L, U, b, x0, ω, iter_max, tol);
    println("El número de iteraciones es de $(info_method[2]).")
    println("El error del método es: $(info_method[1]).")
end