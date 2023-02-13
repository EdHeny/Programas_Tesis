using LinearAlgebra
include("backward_substitution.jl")

function gmres_method(A, b, x0, tol)
    k = size(A,1);
    r0 = b - A*x0;
    β = norm(r0);
    e1 = zeros(k+1);
    e1[1] = 1;
    be1 = β*e1;
    H = zeros(k+1, k);
    Q = zeros(k, k+1);
    Q[:,1] = r0/β;
    xe_sol = ones(dim);
    error_method = 0.0;

    cn = zeros(k);
    sn = zeros(k);
    κ = 0.35;

    for j = 1:k
        vj = A*Q[:,j];
        normvj = norm(vj);
        for i = 1:j
            H[i,j] = dot(vj, Q[:,i]);
            vj = vj - H[i,j]*Q[:,i];
        end
        if norm(vj) / normvj <= κ
            for i = 1:j
                p = dot(Q[:,i],vj);
                vj = vj - p*Q[:,i];
                H[i,j] = H[i,j]+p;
            end
        end
        H[j+1,j] = norm(vj);
        if abs(H[j+1,j]) < tol
            k = j;
            break
        end
        Q[:,j+1] = vj/H[j+1,j];
        for i = 1:j-1
            aux = cn[i]*H[i,j] + sn[i]*H[i+1,j];
            H[i+1,j] = -sn[i]*H[i,j] + cn[i]*H[i+1,j];
            H[i,j] = aux;
        end
        if (abs(H[j,j]) > abs(H[j+1,j]))
            t = H[j+1,j] / H[j,j];
            u = sign(H[j,j]) * sqrt(1 + t^2);
            cn[j] = 1 / u;
            sn[j] = t * cn[j];
        else
            t =  H[j,j] / H[j+1,j];
            u = sign(H[j+1,j]) * sqrt(1 + t^2);
            sn[j] = 1 / u;
            cn[j] = t * sn[j];
        end
        Hjj = H[j,j];
        H[j,j] = cn[j]*Hjj + sn[j]*H[j+1,j];
        H[j+1,j] = -sn[j]*Hjj + cn[j]*H[j+1,j];
        be1[j+1] = -sn[j]*be1[j];
        be1[j] = cn[j]*be1[j];
    end
    for i = 1:k-1
        aux = cn[i]*H[i,k] + sn[i]*H[i+1,k];
        H[i+1,k] = -sn[i]*H[i,k] + cn[i]*H[i+1,k];
        H[i,k] = aux;
    end
    y = backward_substitution(H, be1, k);
    # y = H[1:k,1:k] \ be1[1:k];
    # println(y)
    # println(be1[1:k])
    # println(Q[:,1:k]*y)
    x0 = x0 + Q[:,1:k]*y;
    error_method = norm(x0 - xe_sol) / (norm(x0));
    return x0, error_method, k;
end