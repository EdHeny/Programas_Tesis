using LinearAlgebra
include("backward_substitution.jl")

function gmres_restart_method(A, b, x0, restart, tol)
    m = restart;
    dim = size(A,1);
    # H = Matrix{Float64}(undef,m+1,m);
    # Q = Matrix{Float64}(undef,dim,m+1);
    H = zeros(m+1, m);
    Q = zeros(dim, m+1);
    xp = x0;
    xe_sol = ones(dim);
    error_method = 0.0;

    flag = true;
    k = 0;
    while flag == true
        # println("Reinicio:",k)
        r0 = b - A*xp;
        β = norm(r0);
        Q[:,1] = r0/β;
        e1 = zeros(m+1);
        e1[1] = 1;
        be1 = β*e1;
        cn = zeros(m);
        sn = zeros(m);
        κ = 0.38;
        for j = 1:m
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
                println("fin")
                m = j;
                break
            end
            Q[:,j+1] = vj/H[j+1,j];
            for i = 1:j-1
                aux = H[i,j];
                H[i,j] = cn[i]*aux + sn[i]*H[i+1,j];
                H[i+1,j] = -sn[i]*aux + cn[i]*H[i+1,j];
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
        for i = 1:m-1
            aux = H[i,m];
            H[i,m] = cn[i]*aux + sn[i]*H[i+1,m];
            H[i+1,m] = -sn[i]*aux + cn[i]*H[i+1,m];
        end
        #=
        # y = H[1:m,1:m] \ be1[1:m];
        ym = zeros(m);
        ym[m] = be1[m] / H[m,m];
        for i in m-1:-1:1
            s = dot(H[i,i+1:m],ym[i+1:m]);
            ym[i] = (be1[i] - s) / H[i,i];
        end
        =#
        ym = backward_substitution(H, be1, m);
        xm = xp + Q[:,1:m]*ym;
        error_method = norm(xm - xe_sol) / (norm(xm));
        if error_method < tol
            break;
        end
        xp = xm;
        k += 1;
    end
    return error_method, k;
end