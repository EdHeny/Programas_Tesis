# Implementación de los solvers iterativos JAcobi, Gauss-Seidel y SOR junto con el de CGM.
using LinearAlgebra 
using BenchmarkTools
using Plots

##
# Funciones para construir el sistema Ax = b. Se debe usar un número entero par
# mayor que cero.
# #

##
# Función que construye la matriz "A".
##
function matrizSistema( n )
    M = Tridiagonal(-1*ones(n-1), 3*ones(n), -1*ones(n-1))
    M = M + zeros(n,n)
    for i = 1:n
        if i != n/2 && i != n/2 + 1
            M[i,n+1-i] = 0.5
        end
    end
    return M
end

##
# Función que construye el vector "b".
## 
function vectorSistema(n)
    b = zeros(n,1)
    b[1] = 2.5
    b[n] = 2.5
    for i = 2:n-1
        if i == n/2 || i == n/2+1
            b[i] = 1
        else
            b[i] = 1.5
        end
    end
    return b
end

##
# Funciones que implementan los métodos de Jacobi,
# Gauss-Seidel, SOR y CGM.
##

##
# Implementación del método de Jacobi
##
function metodo_jacobi(D, L, U, x0, iterMax, tol)
    # Norma
    normres_jacobi = []
    x = x0

    for j = 1:iterMax
        x = inv(D)*(b-(L+U)*x)
        normres_jacobi = [normres_jacobi; norm(b-A*x, Inf)]
        if normres_jacobi[end] <= tol
            #println("Jacobi => Iteración máxima: ", j)
            break
        end
    end

    return x, normres_jacobi

end

##
# Implementación del método de Gauss-Seidel
##
function metodo_gs(D, L, U, x0, iterMax, tol)
    # Norma
    normres_gs = []
    x = x0

    for j = 1:iterMax
        x = inv(D+L)*(b-U*x)
        normres_gs = [normres_gs; norm(b-A*x, Inf)]
        if normres_gs[end] <= tol
#            println("GS     => Iteración máxima: ", j)
            break
        end
    end

    return x, normres_gs

end

##
# Implementación del método de SOR
##
function metodo_sor(D, L, U, x0, w, iterMax, tol)
    # Norma
    normres_sor = []
    x = x0

    for j = 1:iterMax
        x = inv(w*L + D)*((1-w)D*x- w*U*x) + w*inv(D + w*L)*b
        normres_sor = [normres_sor; norm(b-A*x, Inf)]
        if normres_sor[end] <= tol
#            println("SOR    => Iteración máxima: ", j)
            break
        end
    end

    return x, normres_sor

end

##
# Implementación del método de CGM
##

function metodo_cgm(A, b, x0, tol)
    x = x0
    r = b - A*x
    d = r
    normres_cgm = []
    for k = 1:length(b)
        rTr = dot(r,r)
        normres_cgm = [normres_cgm; norm(r, Inf)]
        if normres_cgm[end] <= tol
            #println("CGM    => Iteración máxima: ", k)
            break
        end
        alpha = rTr/dot(d,A*d)
        x = x + alpha*d
        r = r - alpha*A*d
        betha = dot(r,r)/rTr
        d = r + betha*d
    end
    return x, normres_cgm
end

##
# Programa de prueba
##

A = matrizSistema( 20 )
b = vectorSistema( 20 )
iterMax = 10000000
tol = 1e-8
w = 1.13

# Descomposición A = D + L + U
D =  Diagonal(A)
L = tril(A,-1)
U = triu(A,1)

# Vector inicial x0
x0 = zeros(size(b))

bchJ = @benchmark xJ, normres_jacobi = metodo_jacobi( D, L, U, x0, iterMax, tol)
#xJ, normres_jacobi = metodo_jacobi( D, L, U, x0, iterMax, tol )
#=
plot(normres_jacobi, color="#6b0851", leg=false)
scatter!(normres_jacobi, label="Jacobi", markersize=4, c="#6b0851", leg=false)
plot!(xaxis=("iterations") , yaxis = ("residuals", :log))
plot!(title="Convergence of Jacobi iteration")
=#

bchGS = @benchmark xG = metodo_gs( D, L, U, x0, iterMax, tol )

bchSOR = @benchmark xS = metodo_sor( D, L, U, x0, w, iterMax, tol )

bchCGM = @benchmark xC = metodo_cgm( A, b, x0, tol)
#=
plot(normres_cgm, color="#6b0851", leg=false)
scatter!(normres_cgm, label="Jacobi", markersize=4, c="#6b0851", leg=false)
plot!(xaxis=("iterations") , yaxis = ("residuals", :log))
plot!(title="Convergence of Jacobi iteration")
=#