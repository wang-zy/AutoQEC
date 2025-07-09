function basisMatrix(fs, tmax, nsteps)
    M = zeros(length(fs),nsteps+1)
    h = tmax/nsteps
    for i = 0:nsteps
        for j = 1:length(fs)
            M[j,i+1] = fs[j](h*i)
        end
    end
    sparse(M)
end

function get_hamiltonian(H0, Hs, u, idx)
    # calculate the Hamiltonian at a given time slice with parameter u
    # H0,Hs must be sparse matrix with the same sparsity pattern.
    # TODO: inplace to be memory efficient
    data = copy(H0.nzval)
    for i = 1:length(Hs)
        @. data += Hs[i].nzval * u[i,idx]
    end
    SparseMatrixCSC(H0.m, H0.n, H0.colptr, H0.rowval, data)
end

function get_dissipator(s, v, idx)
    # v: 1D array storing loss rate for each dissipator in s.As
    # return c_ops, c_dag_ops, cc_ops
    # TODO: inplace to be memory efficient
    c_ops = copy(s.A0)
    for i in 1:length(s.As)
        push!(c_ops, s.As[i]*sqrt(v[i,idx]))
    end
    
    # c_dag_ops = [sparse(c') for c in c_ops]
    c_dag_ops = copy(s.A0_dag)
    for i in 1:length(s.As_dag)
        push!(c_dag_ops, s.As_dag[i]*sqrt(v[i,idx]))
    end

    data = copy(s.A0_dag_A0.nzval)
    for i = 1:length(s.As_dag_As)
        @. data += s.As_dag_As[i].nzval * v[i,idx]
    end
    cc_ops = SparseMatrixCSC(s.A0_dag_A0.m, s.A0_dag_A0.n, s.A0_dag_A0.colptr, s.A0_dag_A0.rowval, data)

    c_ops, c_dag_ops, cc_ops
end

function normalize(psi)
    norm = sum(real(psi).^2) + sum(imag(psi).^2)
    norm, psi/sqrt(norm)
end

function get_pulse(M,theta)
    # M: array (length n_Hs) of basis transformation matrix (shape [n_us, nsteps+1])
    # theta: Array{Float64,1}
    # return pulse: [n_Hs, nsteps+1]
    n_Hs = length(M)
    if n_Hs == 0
        pulse = zeros(0,0)
    else
        nsteps = size(M[1])[2]
        pulse = zeros(n_Hs, nsteps)

        idx = 0
        for i = 1:length(M)
            pulse[i,:] = theta[idx+1:idx+size(M[i])[1]]'*M[i]
            idx += size(M[i])[1]
        end
    end
    pulse
end

function grad_modifier_default(grad)
    # by default the gradients are not modified
    # this function should work IN PLACE
    # overwrite this function by providing the f argument in the training to modify the gradients
end

function default_print(step,F,x...)
    # default printing function
    # first argument is the step number, second argument is the fidelity
    # x... contains the current states and the current gradients (could be different for 1D and 2D)
    # by default only the step number and the current fidelity are printed
    # overwrite this function by providing the g argument in the training to print more as well as save data
    println(step,": ",F)
end

function state_modify_default(psi0,psi1,theta,kappa)
    psi0,psi1,theta,kappa
end

function add_dim(psi, D_old, D_new)
	# D_old: old dimensions
	# D_new: new dimensions
	n = length(D_old)
	idx = zeros(Int64, n)
	psi_new = zeros(CP, prod(D_new))
	for i = 0:length(psi)-1
		x = copy(i)
		for j = n:-1:1
			x,idx[j] = divrem(x,D_old[j])
		end
		new_idx = idx[1]
		for j = 2:n
			new_idx = new_idx * D_new[j] + idx[j]
		end
		psi_new[new_idx+1] = psi[i+1]
	end
	psi_new
end

function meshgrid(xs, ys)
    [xs[i] for i in 1:length(xs), j in 1:length(ys)], [ys[j] for i in 1:length(xs), j in 1:length(ys)]
end

function wigner(rho, xvec, yvec)
    # Using an iterative method to evaluate the wigner functions for the given
    # density matrix.
    M = size(rho)[1]
    X,Y = meshgrid(xvec, yvec)
    A = 0.5 .* sqrt(2) .* (X .+ 1im .* Y)

    Wlist = []
    for i = 1:M
        push!(Wlist, zeros(CP, size(A)))
    end
    @. Wlist[1] = exp(-2.0 * abs(A)^2) / pi

    W = real(rho[1,1]) .* real(Wlist[1])
    for n = 2:M
        @. Wlist[n] = (2 * A * Wlist[n-1]) / sqrt(n-1)
        @. W += 2 * real(rho[1,n] * Wlist[n])
    end

    for m = 1:M-1
        tmp = copy(Wlist[m+1])
        @. Wlist[m+1] = (2 * conj(A) * tmp - sqrt(m) * Wlist[m]) / sqrt(m)
        
        @. W += real(rho[m+1, m+1] * Wlist[m+1])
        
        for n = m+1:M-1
            tmp2 = (2 .* A .* Wlist[n] .- sqrt(m) .* tmp) ./ sqrt(n)
            tmp = copy(Wlist[n+1])
            Wlist[n+1] = copy(tmp2)
            
            @. W += 2 * real(rho[m+1,n+1] * Wlist[n+1])
        end
    end
    W
end

function sparse_common(A, B)
    # return a sparse matrix with data from A and common sparsity pattern of both A and B
    Ia, Ja, Va = findnz(A)
    Ib, Jb, Vb = findnz(B)
    
    na = length(Ia)
    nb = length(Ib)
    
    for i = 1:nb
        append_flag = true
        for j = 1:na
            if (Ia[j], Ja[j]) == (Ib[i], Jb[i])
                append_flag = false
            end
        end
        if append_flag
            append!(Ia, Ib[i])
            append!(Ja, Jb[i])
            append!(Va, 0)
        end
    end

    # hack to force matrix to have the same size.
    append!(Ia,size(A)[1]);
    append!(Ja,size(A)[2]);
    append!(Va,0.0);

    sparse(Ia,Ja,Va)
end

function sparse_H(H0, Hs)
    # return matrices with their original data but common sparsity pattern.
    H0_sp = sparse(H0)
    Hs_sp = [sparse(H) for H in Hs]
    
    for H in Hs_sp
        H0_sp = sparse_common(H0_sp, H)
    end
    Hs_sp = [sparse_common(H, H0_sp) for H in Hs_sp]
    
    H0_sp, Hs_sp
end















function get_avg_fidelity(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    _, psi0 = normalize(psi0)
    _, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    results = pmap(rho -> me_solve(H, c_ops, rho, s.nsteps, s.h), [rho_0, rho_1, rho_10])
    m_rho_0 = results[1]
    m_rho_1 = results[2]
    m_rho_10 = results[3]

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = psi1'*m_rho_0*psi1 + psi0'*m_rho_1*psi0 + 2*real(psi1'*m_rho_10*psi0)

    real(F0/3 + F1/3 + F2/6) # average fidelity
end

####################################temporary!!!##########################################
function me_solve_all(H, c_ops, rho_0, nsteps, h)
    rho_list = zeros(CP, size(rho_0)[1], size(rho_0)[2], nsteps+1)
    c_ops = [sparse(c) for c in c_ops]
    c_dag_ops = [sparse(c') for c in c_ops]
    cc_ops = sparse(sum([c'*c/2 for c in c_ops]))

    # pre-allocate variables
    rho_list[:,:,1] = copy(rho_0)
    k1 = similar(rho_0)
    k2 = similar(rho_0)
    k3 = similar(rho_0)
    k4 = similar(rho_0)
    for i = 1:nsteps
        rho_list[:,:,i+1] = rho_list[:,:,i]
        me_odeint(view(rho_list,:,:,i+1), h, 1, H, c_ops, c_dag_ops, cc_ops, k1,k2,k3,k4)
    end

    rho_list
end

function get_avg_fidelity_all(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    _, psi0 = normalize(psi0)
    _, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    results = pmap(rho -> me_solve_all(H, c_ops, rho, s.nsteps, s.h), [rho_0, rho_1, rho_10])
    m_rho_0 = results[1]
    m_rho_1 = results[2]
    m_rho_10 = results[3]

    F0 = [psi0'*m_rho_0[:,:,i]*psi0 for i = 1:size(m_rho_0)[3]]
    F1 = [psi1'*m_rho_1[:,:,i]*psi1 for i = 1:size(m_rho_0)[3]]
    F2 = [psi1'*m_rho_0[:,:,i]*psi1 + psi0'*m_rho_1[:,:,i]*psi0 + 2*real(psi1'*m_rho_10[:,:,i]*psi0) for i = 1:size(m_rho_0)[3]]

    # real(F0/3 + F1/3 + F2/6) # average fidelity
    real(F0), real(F1), real(F2/2)
end
###############################################################################################
function trace(a,b)
    s = 0.0+0.0im
    for i in 1:size(a)[1]
        for j in 1:size(a)[2]
            s += a[i,j]*b[j,i]
        end
    end
    s
end

function trace(a)
    s = 0.0+0.0im
    for i in 1:size(a)[1]
        s += a[i,i]
    end
    s
end
