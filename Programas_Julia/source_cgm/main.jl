using BenchmarkTools

include("coefmatsys.jl")
include("vecindterm.jl")
include("cg_method.jl")

function main(dim)
    
    A = coefmatsys(dim);
    b = vecindterm(dim);
    xs = zeros(dim);
    tol = 1.0e-10;

    println("Conjugate Gradient Method")
    info_method = cg_method(A, b, xs, tol);
    @time cg_method(A, b, xs, tol);
    println("El número de iteraciones es de $(info_method[2]).")
    println("El error del método es: $(info_method[1]).")

end