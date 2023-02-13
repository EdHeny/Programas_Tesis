using LinearAlgebra

function gmres(A, b, x0, tol)
    k = size(A,1);
    r0 = b - A*x0;
    β = norm(r0);
    e1 = zeros(k+1);
    e1[1] = 1;
    be1 = β*e1;
    H = zeros(k+1, k);
    Q = zeros(k, k+1);
    Q[:,1] = r0/β;

    cn = zeros(k);
    sn = zeros(k);

    for j = 1:k
        vj = A*Q[:,j];
        for i = 1:j
            H[i,j] = dot(vj, Q[:,i]);
            vj = vj - H[i,j]*Q[:,i];
        end
        H[j+1,j] = norm(vj);
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
        hjj = H[j,j];
        H[j,j] = cn[j]*hjj + sn[j]*H[j+1,j];
        H[j+1,j] = -sn[j]*hjj + cn[j]*H[j+1,j];
        #H[j+1,j] = 0.0;
        be1[j+1] = -sn[j]*be1[j]
        be1[j] = cn[j]*be1[j]
        if abs(be1[j+1]) < tol
            k = j;
            break;
        end
    end
    println(be1)
    y = H[1:k,1:k] \ be1[1:k];
    x0 = x0 + Q[:,1:k]*y;
    return x0, H, k;
end