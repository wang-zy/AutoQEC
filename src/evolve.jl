function me(drho, H, rho, c_ops, c_dag_ops, cc_ops, tmp1, tmp2)
    # cc_ops: C^\dagger C/2

    # drho = -1im*(H*rho - rho*H)
    mul!(tmp1, H, rho)
    mul!(tmp2, rho, H)
    @. drho = tmp1 - tmp2

    # drho .+= c_ops[i]*rho*c_dag_ops[i] .- cc_ops[i]*rho .- rho*cc_ops[i]
    for i = 1:length(c_ops)
    	mul!(tmp1, mul!(tmp2, c_ops[i], rho), c_dag_ops[i])
        @. drho += tmp1
    end
    mul!(tmp1, cc_ops, rho)
    mul!(tmp2, rho, cc_ops)
    @. drho -= tmp1 + tmp2
end

function me_odeint(rho, h, nsteps, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4)
    # Master equation time evolution.
    # rho: current state
    # k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    for i = 1:nsteps
        me(k1, H, rho, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k1*h
        me(k2, H, k1, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k2*h^2/2
        me(k1, H, k2, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k1*h^3/6
        me(k2, H, k1, c_ops, c_dag_ops, cc_ops, k3,k4)
        @. rho += k2*h^4/24
    end
end

function me_odeint_td(rho, u, v, s, nsteps, offset, k1,k2,k3,k4,k5,k6,k7)
    # Master equation time evolution.
    # rho: current state
    # k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    h = s.h
    for i = 1:nsteps
        H = get_hamiltonian(s.H0, s.Hs, u, 2*(i+offset)-1)
        c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*(i+offset)-1)
        me(k1, H, rho, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k1*h/2

        H = get_hamiltonian(s.H0, s.Hs, u, 2*(i+offset))
        c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*(i+offset))
        me(k2, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k2*h/2
        me(k3, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k3*h

        H = get_hamiltonian(s.H0, s.Hs, u, 2*(i+offset)+1)
        c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*(i+offset)+1)
        me(k4, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)

        @. rho += k1*h/6 + k2*h/3 + k3*h/3 + k4*h/6
    end
end

