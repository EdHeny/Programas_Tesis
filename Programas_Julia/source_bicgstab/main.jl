using BenchmarkTools

include("coefmatsys.jl")
include("vecindterm.jl")
include("bicgstab_method.jl")

function main(dim)
    
    A = coefmatsys(dim);
    b = vecindterm(dim);
    xs = zeros(dim);
    tol = 1.0e-10;

    println("BiConjugate Gradient Stabilized Method")
    info_method = bicgstab_method(A, b, xs, tol);
    @time bicgstab_method(A, b, xs, tol);
    println("El número de iteraciones es de $(info_method[2]).")
    println("El error del método es: $(info_method[1]).")

end