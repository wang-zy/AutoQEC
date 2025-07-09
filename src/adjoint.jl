function adjoint(da, H, a, c_ops, c_dag_ops, cc_ops, tmp1, tmp2)
    # cc_ops: C^\dagger C/2
    mul!(tmp1, H, a)
    mul!(tmp2, a, H)
    @. da = tmp2 - tmp1
    for i = 1:length(c_ops)
    	mul!(tmp1, mul!(tmp2, c_dag_ops[i], a), c_ops[i])
        @. da += tmp1
    end
    mul!(tmp1, cc_ops, a) # TODO: this would fail if c_ops is empty!!!
    mul!(tmp2, a, cc_ops)
    @. da -= tmp1 + tmp2
end

function overlap(a, Delta)
    s = 0.0
    for n = 1:size(Delta)[2], m = 1:size(Delta)[1]
         s += real(Delta[m,n])*real(a[m,n]) + imag(Delta[m,n])*imag(a[m,n])
    end
    s
end

function update_a_theta(a_theta, s, rho, a, h, k1,k2)
    for j = 1:length(s.Hs)
        mul!(k1, s.Hs_fast[j], rho)
        mul!(k2, rho, s.Hs_fast[j])
        @. k1 -= k2
        a_theta[j] += overlap(a, k1) * h
    end
end

function update_a_kappa(a_kappa, s, rho, a, h, k1,k2,k3)
    for i = 1:length(s.As)
        mul!(k1, mul!(k2, s.As[i], rho), s.As_dag[i])
        mul!(k2, s.As_dag_As_fast[i], rho)
        mul!(k3, rho, s.As_dag_As_fast[i])
        @. k1 -= k2 + k3
        a_kappa[i] += overlap(a, k1) * h
    end
end

function adjoint_odeint(a, rho, a_theta, a_kappa, s, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4,k5,k6,k7,k8)
    # adjoint equation time evolution.
    # state_new: current state
    # state_tmp, k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    h = s.h

    # calculate all rho related quantities
    k1 = copy(rho)
    k2 = copy(rho)
    k3 = copy(rho)
    me(k4, H, rho, c_ops, c_dag_ops, cc_ops, k7,k8) # first derivative
    me(k5, H, k4, c_ops, c_dag_ops, cc_ops, k7,k8) # second derivative
    me(k6, H, k5, c_ops, c_dag_ops, cc_ops, k7,k8) # third derivative
    @. k1 -= k4*h/2 - k5*h^2/6 + k6*h^3/24
    @. k2 -= k4*h*2/3 - k5*h^2/4
    @. k3 -= k4*h*3/4

    update_a_theta(a_theta, s, k1, a, h, k7,k8)
    update_a_kappa(a_kappa, s, k1, a, h, k6,k7,k8)

    adjoint(k1, H, a, c_ops, c_dag_ops, cc_ops, k7,k8)
    update_a_theta(a_theta, s, k2, k1, h^2/2, k7,k8)
    update_a_kappa(a_kappa, s, k2, k1, h^2/2, k6,k7,k8)

    adjoint(k2, H, k1, c_ops, c_dag_ops, cc_ops, k7,k8)
    update_a_theta(a_theta, s, k3, k2, h^3/6, k7,k8)
    update_a_kappa(a_kappa, s, k3, k2, h^3/6, k6,k7,k8)

    adjoint(k3, H, k2, c_ops, c_dag_ops, cc_ops, k7,k8)
    update_a_theta(a_theta, s, rho, k3, h^4/24, k7,k8)
    update_a_kappa(a_kappa, s, rho, k3, h^4/24, k6,k7,k8)

    adjoint(k4, H, k3, c_ops, c_dag_ops, cc_ops, k7,k8)
    @. a += k1*h + k2*h^2/2 + k3*h^3/6 + k4*h^4/24
end

function me_adjoint_solve(H, c_ops, rho_0, a_T, s, theta, kappa, f)
    rho_T = similar(rho_0)
    a_theta = zeros(length(theta))
    a_kappa = zeros(length(kappa))

    c_ops = [sparse(c) for c in c_ops]
    c_dag_ops = [sparse(c') for c in c_ops]
    cc_ops = sparse(sum([c'*c/2 for c in c_ops]))

    # checkpoints
    cpts = zeros(CP, size(a_T)[1], size(a_T)[2], s.ncpts)
    cpts[:,:,1] = rho_0

    # pre-allocate variables
    rho = similar(rho_0)
    a = copy(a_T)
    k1 = similar(a_T)
    k2 = similar(a_T)
    k3 = similar(a_T)
    k4 = similar(a_T)
    k5 = similar(a_T)
    k6 = similar(a_T)
    k7 = similar(a_T)
    k8 = similar(a_T)

    flag = true # whether or not store rho_T; only true for the first time reaching rho_T

    for step in s.seq
        start_cpt_idx = step[1]
        end_cpt_idx = step[3]
        nsteps = step[2]
        if end_cpt_idx == -1
            for n = nsteps:-1:1
                # forward time evolution
                @. rho = cpts[:,:,start_cpt_idx]
                me_odeint(rho, s.h, n, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4)
                if flag
                    # save the final evolved state, only run once
                    @. rho_T = rho
                    flag = false
                    f(a, rho_T) # modify the adjoint based on the final state
                end
                # calculate adjoint backward
                adjoint_odeint(a, rho, a_theta, a_kappa, s, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4,k5,k6,k7,k8)
            end
        else
            # forward evolve by nsteps
            cpts[:,:,end_cpt_idx] = cpts[:,:,start_cpt_idx]
            me_odeint(view(cpts,:,:,end_cpt_idx), s.h, nsteps, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4)
        end
    end

    rho_T, a, a_theta, a_kappa
end

function adjoint_odeint_td(a, rho, s, u, v, idx, k1,k2,k3,k4,k5,k6,k7)
    # adjoint equation time evolution.
    # state_new: current state
    # state_tmp, k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    h = s.h
    H = get_hamiltonian(s.H0, s.Hs, u, 2*idx-1)
    c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*idx-1)
    adjoint(k1, H, a, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k1*h/2

    H = get_hamiltonian(s.H0, s.Hs, u, 2*idx-2)
    c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*idx-2)
    adjoint(k2, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k2*h/2
    adjoint(k3, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k3*h

    H = get_hamiltonian(s.H0, s.Hs, u, 2*idx-3)
    c_ops, c_dag_ops, cc_ops = get_dissipator(s, v, 2*idx-3)
    adjoint(k4, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)

    @. a += k1*h/6 + k2*h/3 + k3*h/3 + k4*h/6
end

function me_adjoint_solve_td(u, v, rho_0, a_T, s, theta, kappa, f)
    # u/v: pulse shape for Hs/As with size[n_Hs/n_As,nsteps+1]
    rho_T = similar(rho_0)
    a_theta = zeros(length(theta))
    a_kappa = zeros(length(kappa))

    # checkpoints
    cpts = zeros(CP, size(a_T)[1], size(a_T)[2], s.ncpts)
    cpts[:,:,1] = rho_0

    # pre-allocate variables
    rho = similar(rho_0)
    a = copy(a_T)
    k1 = similar(a_T)
    k2 = similar(a_T)
    k3 = similar(a_T)
    k4 = similar(a_T)
    k5 = similar(a_T)
    k6 = similar(a_T)
    k7 = similar(a_T)

    flag = true # whether or not store rho_T; only true for the first time reaching rho_T

    overlap_theta = zeros(length(s.Hs),s.nsteps+1)
    overlap_kappa = zeros(length(s.As),s.nsteps+1)

    for step in s.seq
        start_cpt_idx = step[1]
        end_cpt_idx = step[3]
        nsteps = step[2]
        start_idx = step[4] # time index for the starting checkpoint, range [0,nsteps]
        if end_cpt_idx == -1
            for n = nsteps:-1:1
                # forward time evolution
                @. rho = cpts[:,:,start_cpt_idx]
                me_odeint_td(rho, u, v, s, n, start_idx, k1,k2,k3,k4,k5,k6,k7)
                if flag
                    # save the final evolved state, only run once
                    @. rho_T = rho
                    flag = false
                    f(a, rho_T) # modify the adjoint based on the final state
                end
                # update overlaps
                update_a_theta(view(overlap_theta,:,start_idx+n+1), s, rho, a, s.h, k1,k2)
                update_a_kappa(view(overlap_kappa,:,start_idx+n+1), s, rho, a, s.h, k1,k2,k3)

                # calculate adjoint backward
                adjoint_odeint_td(a, rho, s, u, v, start_idx+n+1, k1,k2,k3,k4,k5,k6,k7)
            end
        else
            # forward evolve by nsteps
            cpts[:,:,end_cpt_idx] = cpts[:,:,start_cpt_idx]
            me_odeint_td(view(cpts,:,:,end_cpt_idx), u, v, s, nsteps, start_idx, k1,k2,k3,k4,k5,k6,k7)
        end
    end
    # update overlaps
    update_a_theta(view(overlap_theta,:,1), s, rho_0, a, s.h, k1,k2)
    update_a_kappa(view(overlap_kappa,:,1), s, rho_0, a, s.h, k1,k2,k3)

    # calculate adjoint gradients a_theta, a_kappa from the overlaps
    # Mu/Mv size [n_us/n_vs, 2*nsteps+1]
    kk = 0
    for i = 1:length(s.Mu)
        n_us = size(s.Mu[i])[1]
        for j = 1:n_us
            for k = div(s.nsteps,2):-1:1
                a_theta[kk+j] += (overlap_theta[i,k*2+1]*s.Mu[i][j,k*4+1]+4*overlap_theta[i,k*2]*s.Mu[i][j,k*4-1]+overlap_theta[i,k*2-1]*s.Mu[i][j,k*4-3])/3
            end
        end
        kk += n_us
    end
    kk = 0
    for i = 1:length(s.Mv)
        n_vs = size(s.Mv[i])[1]
        for j = 1:n_vs
            for k = div(s.nsteps,2):-1:1
                a_kappa[kk+j] += (overlap_kappa[i,k*2+1]*s.Mv[i][j,k*4+1]+4*overlap_kappa[i,k*2]*s.Mv[i][j,k*4-1]+overlap_kappa[i,k*2-1]*s.Mv[i][j,k*4-3])/3
            end
        end
        kk += n_vs
    end

    rho_T, a, a_theta, a_kappa
end





















function update_a_theta_new(a_theta, s, rho, a, k1,k2)
    for j = 1:length(s.Hs)
        mul!(k1, s.Hs_fast[j], rho)
        mul!(k2, rho, s.Hs_fast[j])
        @. k1 -= k2
        a_theta[j] += overlap(a, k1)
    end
end

function update_a_kappa_new(a_kappa, s, rho, a, k1,k2,k3)
    for i = 1:length(s.As)
        mul!(k1, mul!(k2, s.As[i], rho), s.As_dag[i])
        mul!(k2, s.As_dag_As_fast[i], rho)
        mul!(k3, rho, s.As_dag_As_fast[i])
        @. k1 -= k2 + k3
        a_kappa[i] += overlap(a, k1)
    end
end

function get_hamiltonian_new(s,lambda,t)
    # calculate the Hamiltonian at a given time slice with parameter u
    # H0,Hs must be sparse matrix with the same sparsity pattern.
    # TODO: inplace to be memory efficient
    data = copy(s.H0.nzval)
    if length(s.Hs) != 0
        s.uus(lambda,t,s.u)
    end
    for i = 1:length(s.Hs)
        # println(i)
        # println(s.Hs[i].nzval * s.uus[i](lambda,t))
        # @. data += s.Hs[i].nzval * s.uus[i](lambda,t)
        # data += s.Hs[i].nzval * s.uus[i](lambda,t)
        # @. data += s.Hs[i].nzval * u[i]
        @. data += s.Hs[i].nzval * s.u[i]
        # println("here")
    end
    SparseMatrixCSC(s.H0.m, s.H0.n, s.H0.colptr, s.H0.rowval, data)
end

function get_dissipator_new(s,lambda,t)
    # v: 1D array storing loss rate for each dissipator in s.As
    # return c_ops, c_dag_ops, cc_ops
    # TODO: inplace to be memory efficient
    c_ops = copy(s.A0)
    if length(s.As) != 0
        s.vvs(lambda,t,s.v)
    end
    for i in 1:length(s.As)
        # push!(c_ops, s.As[i]*sqrt(s.vvs[i](lambda,t)))
        # push!(c_ops, s.As[i]*sqrt(v[i]))
        push!(c_ops, s.As[i]*sqrt(s.v[i]))
    end
    
    # c_dag_ops = [sparse(c') for c in c_ops]
    c_dag_ops = copy(s.A0_dag)
    for i in 1:length(s.As_dag)
        # push!(c_dag_ops, s.As_dag[i]*sqrt(s.vvs[i](lambda,t)))
        # push!(c_dag_ops, s.As_dag[i]*sqrt(v[i]))
        push!(c_dag_ops, s.As_dag[i]*sqrt(s.v[i]))
    end

    data = copy(s.A0_dag_A0.nzval)
    for i = 1:length(s.As_dag_As)
        # @. data += s.As_dag_As[i].nzval * s.vvs[i](lambda,t)
        # data += s.As_dag_As[i].nzval * s.vvs[i](lambda,t)
        # @. data += s.As_dag_As[i].nzval * v[i]
        @. data += s.As_dag_As[i].nzval * s.v[i]
    end
    cc_ops = SparseMatrixCSC(s.A0_dag_A0.m, s.A0_dag_A0.n, s.A0_dag_A0.colptr, s.A0_dag_A0.rowval, data)

    c_ops, c_dag_ops, cc_ops
end

function me_odeint_td_new(rho, s, lambda, nsteps, offset, k1,k2,k3,k4,k5,k6,k7)
    # Master equation time evolution.
    # rho: current state
    # k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    h = s.h
    for i = 1:nsteps
        t = h*(i+offset-1)
        H = get_hamiltonian_new(s,lambda,t)
        c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t)
        me(k1, H, rho, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k1*h/2

        H = get_hamiltonian_new(s,lambda,t+h/2)
        c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t+h/2)
        me(k2, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k2*h/2
        me(k3, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
        @. k5 = rho + k3*h

        H = get_hamiltonian_new(s,lambda,t+h)
        c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t+h)
        me(k4, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)

        @. rho += k1*h/6 + k2*h/3 + k3*h/3 + k4*h/6
    end
end

function adjoint_odeint_td_new(a, rho, s, lambda, idx, k1,k2,k3,k4,k5,k6,k7)
    # adjoint equation time evolution.
    # state_new: current state
    # state_tmp, k1-k4: pre-allocated temporary variables for memory efficiency
    # h: integration time step
    h = s.h
    t = (idx-1)*h
    H = get_hamiltonian_new(s,lambda,t)
    c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t)
    adjoint(k1, H, a, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k1*h/2

    H = get_hamiltonian_new(s,lambda,t-h/2)
    c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t-h/2)
    adjoint(k2, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k2*h/2
    adjoint(k3, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)
    @. k5 = a + k3*h

    H = get_hamiltonian_new(s,lambda,t-h)
    c_ops, c_dag_ops, cc_ops = get_dissipator_new(s,lambda,t-h)
    adjoint(k4, H, k5, c_ops, c_dag_ops, cc_ops, k6,k7)

    @. a += k1*h/6 + k2*h/3 + k3*h/3 + k4*h/6
end

function me_adjoint_solve_td_new(rho_0, a_T, s, lambda, f)
    rho_T = similar(rho_0)
    a_lambda = zeros(length(lambda))

    # checkpoints
    cpts = zeros(CP, size(a_T)[1], size(a_T)[2], s.ncpts)
    cpts[:,:,1] = rho_0

    # pre-allocate variables
    rho = similar(rho_0)
    a = copy(a_T)
    k1 = similar(a_T)
    k2 = similar(a_T)
    k3 = similar(a_T)
    k4 = similar(a_T)
    k5 = similar(a_T)
    k6 = similar(a_T)
    k7 = similar(a_T)

    flag = true # whether or not store rho_T; only true for the first time reaching rho_T

    overlap_theta = zeros(length(s.Hs),s.nsteps+1)
    overlap_kappa = zeros(length(s.As),s.nsteps+1)

    for step in s.seq
        start_cpt_idx = step[1]
        end_cpt_idx = step[3]
        nsteps = step[2]
        start_idx = step[4] # time index for the starting checkpoint, range [0,nsteps]
        if end_cpt_idx == -1
            for n = nsteps:-1:1
                # forward time evolution
                @. rho = cpts[:,:,start_cpt_idx]
                me_odeint_td_new(rho, s, lambda, n, start_idx, k1,k2,k3,k4,k5,k6,k7)
                if flag
                    # save the final evolved state, only run once
                    @. rho_T = rho
                    flag = false
                    f(a, rho_T) # modify the adjoint based on the final state
                end
                # update overlaps
                update_a_theta_new(view(overlap_theta,:,start_idx+n+1), s, rho, a, k1,k2)
                update_a_kappa_new(view(overlap_kappa,:,start_idx+n+1), s, rho, a, k1,k2,k3)

                # calculate adjoint backward
                adjoint_odeint_td_new(a, rho, s, lambda, start_idx+n+1, k1,k2,k3,k4,k5,k6,k7)
            end
        else
            # forward evolve by nsteps
            cpts[:,:,end_cpt_idx] = cpts[:,:,start_cpt_idx]
            me_odeint_td_new(view(cpts,:,:,end_cpt_idx), s, lambda, nsteps, start_idx, k1,k2,k3,k4,k5,k6,k7)
        end
    end
    # update overlaps
    update_a_theta_new(view(overlap_theta,:,1), s, rho_0, a, k1,k2)
    update_a_kappa_new(view(overlap_kappa,:,1), s, rho_0, a, k1,k2,k3)

    # calculate adjoint gradients a_theta, a_kappa from the overlaps
    for k = s.nsteps+1:-1:1
        t = (k-1)*s.h
        tmp = zeros(length(lambda))
        if length(s.Hs) != 0
            tmp += s.us_grad(lambda,t)*overlap_theta[:,k]
        end
        if length(s.As) != 0
            tmp += s.vs_grad(lambda,t)*overlap_kappa[:,k]
        end

        if k == 1 || k == s.nsteps+1
            a_lambda += tmp
        elseif k%2 == 0
            a_lambda += 4*tmp
        else
            a_lambda += 2*tmp
        end
    end
    
    rho_T, a, a_lambda*s.h/3
end
