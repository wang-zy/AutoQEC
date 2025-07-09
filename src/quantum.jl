function id(N)
    a=[(i==j)*1.0 for i in 1:N, j in 1:N]+ (0.0+1im*0.0)*zeros(N,N)
end

function destroy(N)
    a=(0.0+1im*0.0)*zeros(N,N)
    a=[(i==j-1)*sqrt(i) for i in 1:N, j in 1:N]+ (0.0+1im*0.0)*zeros(N,N)
end

function basis(N, j)
    psi = zeros(CP, N)
    psi[j+1] = 1.0
    psi
end

function coherent(N, alpha)
    psi = zeros(CP, N)
    for n in 0:N-1
        psi[n+1] = exp(-abs(alpha)^2/2) * float(alpha)^n / sqrt(factorial(big(n)))
    end
    psi
end

function tensor(ops)
    op = ops[1]
    for i = 2:length(ops)
        op = kron(op,ops[i])
    end
    op
end

function displace(N,alpha)
    a = destroy(N)
    exp(alpha*a'-alpha'*a)
end

# qubit related functions
# notice that ground state is basis(2,1) while excited state is basis(N,0)
function sigmax()
    [0 1 ; 1 0.0im]
end

function sigmay()
    [0 -1im ; 1.0im 0]
end

function sigmaz()
    [1 0.0im ; 0 -1]
end

function sigmam()
    [0 0 ; 1 0.0im]
end

function sigmap()
    [0 1 ; 0 0.0im]
end