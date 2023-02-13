using BenchmarkTools

include("coefmatsys.jl")
include("vecindterm.jl")
include("gmres_method.jl")
# include("gmres2.jl")
include("gmres_restart_method.jl")

dim = 60;
A = coefmatsys(dim);
b = vecindterm(dim);
xs = zeros(dim);
restart = 10;
tol = 1.0e-10;

println("Generelized Minimal Residual Method")
info_method = gmres_method(A, b, xs, tol);
@time gmres_method(A, b, xs, tol);
println("El número de iteraciones es de $(info_method[3]).")
println("El error del método es: $(info_method[2]).")
println("---------------------------------------------------------")
println("Generelized Minimal Residual Restarted Method")
info_method = gmres_restart_method(A, b, xs, restart, tol);
@time gmres_restart_method(A, b, xs, restart, tol);
println("El número de reinicios es de $(info_method[2]).")
println("El error del método es: $(info_method[1]).")
# println("El tiempo de ejecución es: $(minimum(t)).")