function me_solve(H0, Hs, theta, c_ops, rho_0, nsteps, tmax)
    h = tmax / nsteps
    H = copy(H0)
    for j in 1:length(theta)
        H += Hs[j]*theta[j]
    end

    c_ops = [sparse(c) for c in c_ops]
    c_dag_ops = [sparse(c') for c in c_ops]
    cc_ops = sparse(sum([c'*c/2 for c in c_ops]))

    # pre-allocate variables
    rho = copy(rho_0)
    k1 = similar(rho_0)
    k2 = similar(rho_0)
    k3 = similar(rho_0)
    k4 = similar(rho_0)
    me_odeint(rho, h, nsteps, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4)

    rho
end

function me_solve_td(H0, Hs, us, theta, c_ops, rho_0, nsteps, tmax)
    Mu = [basisMatrix(us[i], tmax, nsteps) for i = 1:length(us)]
    u = get_pulse(Mu,theta)
    c_ops = [sparse(c) for c in c_ops]
    c_dag_ops = [sparse(c') for c in c_ops]
    cc_ops = sparse(sum([c'*c/2 for c in c_ops]))

    # pre-allocate variables
    rho = copy(rho_0)
    k1 = similar(rho_0)
    k2 = similar(rho_0)
    k3 = similar(rho_0)
    k4 = similar(rho_0)

    rho_list = zeros(CP, size(rho)[1], size(rho)[2], nsteps+1)
    rho_list[:,:,1] = rho_0

    h = tmax/nsteps
    for i = 1:nsteps
        H = -1.0im*(H0 + sum([Hs[j]*u[j,i] for j = 1:length(Hs)]))
        # H = get_hamiltonian(H0, Hs, u, i)
        # c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, i)
        
        me(k1, H, rho, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k1*h
        me(k2, H, k1, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k2*h^2/2
        me(k1, H, k2, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k1*h^3/6
        me(k2, H, k1, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k2*h^4/24

        rho_list[:,:,i+1] = rho
    end

    rho_list
end
