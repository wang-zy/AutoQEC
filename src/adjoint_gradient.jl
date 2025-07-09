#####################################################################################
#
#                          Generic training functions
#
# Description: logical qubit means learning a 2D subspace
#              state stabilization means learning a 1D subspace
function learning_logical_qubit(s::setting,psi0,psi1,adam,niters;
                                theta=Float64[],kappa=Float64[],theta_lb=Float64[],
                                theta_ub=Float64[],kappa_lb=Float64[],kappa_ub=Float64[],
                                f=grad_modifier_default,g=default_print,h=state_modify_default,
                                grad_f=logical_qubit_average_fidelity)
    # reset the internal state of the adam optimizer
    ResetAdam(adam)

    # convert theta and kappa to angle variables to optimizer over
    alpha = similar(theta)
    beta = similar(kappa)
    @. alpha = asin((2*theta - theta_ub - theta_lb)/(theta_ub - theta_lb))
    @. beta = asin((2*kappa - kappa_ub - kappa_lb)/(kappa_ub - kappa_lb))

    # pre-allocate the gradients for alpha and beta
    grad_alpha = similar(theta)
    grad_beta = similar(kappa)

    x = pack(psi0, psi1, alpha, beta)

    F_list = Float64[]

    for step in 1:niters
        F, grad_psi0, grad_psi1, grad_theta, grad_kappa = grad_f(s,psi0,psi1,theta=theta,kappa=kappa)
        push!(F_list,F)

        # print and save stuff (before the update step)
        g(step, F, psi0, psi1, theta, kappa, grad_psi0, grad_psi1, grad_theta, grad_kappa, F_list)

        @. grad_alpha = (theta_ub - theta_lb)/2 * cos(alpha) * grad_theta
        @. grad_beta = (kappa_ub - kappa_lb)/2 * cos(beta) * grad_kappa

        grad = pack(grad_psi0, grad_psi1, grad_alpha, grad_beta)
        f(grad) # modify the gradients in place; by default f is identity mapping
        GradAscentAdam(adam, x, grad, step)
        unpack(x, psi0, psi1, alpha, beta)

        @. theta = (theta_ub + theta_lb)/2 + (theta_ub - theta_lb)/2 * sin(alpha)
        @. kappa = (kappa_ub + kappa_lb)/2 + (kappa_ub - kappa_lb)/2 * sin(beta)

        # maintain orthogonality between psi0 and psi1
        r = (psi0'*psi1)/(psi0'*psi0)
        @. psi1 -= r * psi0

        psi0,psi1,theta,kappa = h(psi0,psi1,theta,kappa)
    end
    F_list
end

function learning_state_stabilization(s::setting,psi0,adam,niters;
                                      theta=Float64[],kappa=Float64[],theta_lb=Float64[],
                                      theta_ub=Float64[],kappa_lb=Float64[],kappa_ub=Float64[],
                                      f=grad_modifier_default,g=default_print,
                                      grad_f=state_stabilization_fidelity)
    # reset the internal state of the adam optimizer
    ResetAdam(adam)

    # convert theta and kappa to angle variables to optimizer over
    alpha = similar(theta)
    beta = similar(kappa)
    @. alpha = asin((2*theta - theta_ub - theta_lb)/(theta_ub - theta_lb))
    @. beta = asin((2*kappa - kappa_ub - kappa_lb)/(kappa_ub - kappa_lb))

    # pre-allocate the gradients for alpha and beta
    grad_alpha = similar(theta)
    grad_beta = similar(kappa)

    x = pack(psi0, alpha, beta)

    F_list = Float64[]

    for step in 1:niters
        F, grad_psi0, grad_theta, grad_kappa = grad_f(s,psi0,theta=theta,kappa=kappa)
        push!(F_list,F)

        # print and save stuff (before the update step)
        g(step, F, psi0, theta, kappa, grad_psi0, grad_theta, grad_kappa, F_list)

        @. grad_alpha = (theta_ub - theta_lb)/2 * cos(alpha) * grad_theta
        @. grad_beta = (kappa_ub - kappa_lb)/2 * cos(beta) * grad_kappa

        grad = pack(grad_psi0, grad_alpha, grad_beta)
        f(grad) # modify the gradients in place; by default f is identity mapping
        GradAscentAdam(adam, x, grad, step)
        unpack(x, psi0, alpha, beta)

        @. theta = (theta_ub + theta_lb)/2 + (theta_ub - theta_lb)/2 * sin(alpha)
        @. kappa = (kappa_ub + kappa_lb)/2 + (kappa_ub - kappa_lb)/2 * sin(beta)
    end
    F_list
end


# temporary, fix later
function learning_state_stabilization_map(s::setting,psi0,psi1,psi2,adam,niters;
                                      theta=Float64[],kappa=Float64[],theta_lb=Float64[],
                                      theta_ub=Float64[],kappa_lb=Float64[],kappa_ub=Float64[],
                                      f=grad_modifier_default,g=default_print,
                                      grad_f=state_stabilization_map_fidelity)
    # learn the mapping from psi0 to psi1, hopefully psi1 is being stabilized
    # update: psi2 for tracing over lossy qubit < quick hack, fix later
    # reset the internal state of the adam optimizer
    ResetAdam(adam)

    # convert theta and kappa to angle variables to optimizer over
    alpha = similar(theta)
    beta = similar(kappa)
    @. alpha = asin((2*theta - theta_ub - theta_lb)/(theta_ub - theta_lb))
    @. beta = asin((2*kappa - kappa_ub - kappa_lb)/(kappa_ub - kappa_lb))

    # pre-allocate the gradients for alpha and beta
    grad_alpha = similar(theta)
    grad_beta = similar(kappa)

    x = pack(psi0, alpha, beta)

    F_list = Float64[]

    for step in 1:niters
        F, grad_psi0, grad_theta, grad_kappa = grad_f(s,psi0,psi1,psi2,theta=theta,kappa=kappa)
        push!(F_list,F)

        # print and save stuff (before the update step)
        g(step, F, psi0, theta, kappa, grad_psi0, grad_theta, grad_kappa, F_list)

        @. grad_alpha = (theta_ub - theta_lb)/2 * cos(alpha) * grad_theta
        @. grad_beta = (kappa_ub - kappa_lb)/2 * cos(beta) * grad_kappa

        grad = pack(grad_psi0, grad_alpha, grad_beta)
        f(grad) # modify the gradients in place; by default f is identity mapping
        GradAscentAdam(adam, x, grad, step)
        unpack(x, psi0, alpha, beta)

        @. theta = (theta_ub + theta_lb)/2 + (theta_ub - theta_lb)/2 * sin(alpha)
        @. kappa = (kappa_ub + kappa_lb)/2 + (kappa_ub - kappa_lb)/2 * sin(beta)
    end
    F_list
end

# temporary, fix later
function state_stabilization_map_fidelity(s::setting,psi0,psi1,psi2;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0: state to be stabilized

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)    
    rho_0 = psi0*psi0'

    rho_a = psi1*psi1'+psi2*psi2'

    function f(a,rho)
    end

    m_rho_0, ma_rho_0, a_theta, a_kappa = me_adjoint_solve(H, c_ops, rho_0, rho_a, s, theta, kappa,f)

    F = real(psi1'*m_rho_0*psi1) + real(psi2'*m_rho_0*psi2)

    # Calculate gradients
    A = m_rho_0 + ma_rho_0    
    grad0 = ((A+A')*psi0 - 4*F*psi0)/2/norm_0
    
    # F, grad0/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, a_theta, a_kappa
end




#####################################################################################
#
#                 Gradient functions for different loss functions
#
# Description: calculate the gradients with respect to the states as well as 
#              system parameters using the adjoint gradient method.
#
# General form: grad_f_2D(s::setting,psi0,psi1,theta,kappa)
#               grad_f_1D(s::setting,psi0,theta,kappa)
# available options:
#     - logical_qubit_average_fidelity
#     - logical_qubit_entanglement_fidelity
#     - biased_logical_qubit_average_fidelity
#     - biased_logical_qubit_trace_distance
#     - state_stabilization_fidelity
#     - state_stabilization_trace_distance
#     - logical_qubit_average_fidelity_td
#     - logical_qubit_entanglement_fidelity_td
#     - biased_logical_qubit_average_fidelity_td
#     - biased_logical_qubit_trace_distance_td
#     - state_stabilization_fidelity_td
#     - state_stabilization_trace_distance_td
# see function documentation below for more information.

function logical_qubit_average_fidelity(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
	# Adjoint method for learning the logical qubit subspace
	# Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
	# Parameters:
	#     - H0: constant part of the Hamiltonian
	#     - Hs: [H_j, j=1,2,...]
	#     - theta: [theta_j, j=1,2,...]
	#     - A0: list of the constant dissipators
	#     - As: [A_i, i==1,2,...]
	#     - kappa: [kappa_i, i=1,2,...]
	#     - psi0, psi1: current basis vectors of the logical subspace

	# get Hamiltonian and dissipators
	# H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
	H = copy(s.H0)
	for j in 1:length(theta)
		H += s.Hs[j]*theta[j]
	end
	# c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
	c_ops = copy(s.A0)
	for i in 1:length(kappa)
		push!(c_ops, s.As[i]*sqrt(kappa[i]))
	end

	norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    function f1(a,rho)
    end
    # function f2(a,rho)
    # end
    function f2(a,rho)
        tmp = trace(a', rho)
        tmp = tmp / abs(tmp)
        @. a *= tmp
    end

    results = pmap((rho,a,f) -> me_adjoint_solve(H, c_ops, rho, a, s, theta, kappa,f), [rho_0, rho_1, rho_10], [rho_0+0.5*rho_1, rho_1+0.5*rho_0, rho_10],[f1,f1,f2])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]
    m_rho_10, ma_rho_10, a_theta_10, a_kappa_10 = results[3]

    ttt = psi1'*m_rho_10*psi0

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = psi1'*m_rho_0*psi1 + psi0'*m_rho_1*psi0 + 2*abs(ttt)
    # F2 = psi1'*m_rho_0*psi1 + psi0'*m_rho_1*psi0 + 2*real(ttt)

    F = real(F0/3 + F1/3 + F2/6) # average fidelity

    a_theta = (a_theta_0 + a_theta_1 + a_theta_10)/3
    a_kappa = (a_kappa_0 + a_kappa_1 + a_kappa_10)/3

    # Calculate gradients
    A = m_rho_0 + ma_rho_0*4.0/3.0 - ma_rho_1*2.0/3.0
    B = m_rho_1 + ma_rho_1*4.0/3.0 - ma_rho_0*2.0/3.0
    C = m_rho_10*ttt'/abs(ttt) + ma_rho_10
    
    g1 = (A+A')*psi0 - 4*F0*psi0
    g2 = (B+B')*psi0 + 2*conj(psi1'*C)[:] - 2*F2*psi0
    grad0 = (2*g1 + g2)/6/norm_0
    
    g1 = (B+B')*psi1 - 4*F1*psi1
    g2 = (A+A')*psi1 + 2*C*psi0 - 2*F2*psi1
    grad1 = (2*g1 + g2)/6/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end

function logical_qubit_entanglement_fidelity(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    function f1(a,rho)
    end
    function f2(a,rho)
        tmp = trace(a', rho)
        tmp = tmp / abs(tmp)
        @. a *= tmp
    end

    results = pmap((rho,a,f) -> me_adjoint_solve(H, c_ops, rho, a, s, theta, kappa,f), [rho_0, rho_1, rho_10], [rho_0, rho_1, rho_10],[f1,f1,f2])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]
    m_rho_10, ma_rho_10, a_theta_10, a_kappa_10 = results[3]

    ttt = psi1'*m_rho_10*psi0

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = 2*abs(ttt)

    F = real(F0 + F1 + F2)/4 # entanglement fidelity

    a_theta = (a_theta_0 + a_theta_1 + 2*a_theta_10)/4
    a_kappa = (a_kappa_0 + a_kappa_1 + 2*a_kappa_10)/4

    # Calculate gradients
    A = m_rho_0 + ma_rho_0
    B = m_rho_1 + ma_rho_1
    C = m_rho_10*ttt'/abs(ttt) + ma_rho_10
    
    g1 = (A+A')*psi0 - 4*F0*psi0
    g2 = 2*conj(psi1'*C)[:] - 2*F2*psi0
    grad0 = (g1 + 2*g2)/4/norm_0
    
    g1 = (B+B')*psi1 - 4*F1*psi1
    g2 = 2*C*psi0 - 2*F2*psi1
    grad1 = (g1 + 2*g2)/4/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end

function biased_logical_qubit_average_fidelity(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'

    function f(a,rho)
    end

    results = pmap((rho,a,f) -> me_adjoint_solve(H, c_ops, rho, a, s, theta, kappa,f), [rho_0, rho_1], [rho_0, rho_1],[f,f])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1

    F = real(F0+F1)/2 # average fidelity

    a_theta = (a_theta_0 + a_theta_1)/2
    a_kappa = (a_kappa_0 + a_kappa_1)/2

    # Calculate gradients
    A = m_rho_0 + ma_rho_0
    B = m_rho_1 + ma_rho_1
    
    grad0 = ((A+A')*psi0 - 4*F0*psi0)/2/norm_0
    grad1 = ((B+B')*psi1 - 4*F1*psi1)/2/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end

function biased_logical_qubit_trace_distance(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'

    function f(a,rho_T)
        # here a must be a copy of the initial state rho_0
        S = svd(rho_T-a)
        mul!(a, S.U, S.Vt)
    end

    results = pmap((rho,a,f) -> me_adjoint_solve(H, c_ops, rho, a, s, theta, kappa,f), [rho_0, rho_1], [rho_0, rho_1],[f,f])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]

    S0 = svd(m_rho_0 - rho_0)
    S1 = svd(m_rho_1 - rho_1)

    F0 = sum(S0.S)
    F1 = sum(S1.S)

    F = real(F0+F1)/4 # average trace distance

    F_avg = real(psi0'*m_rho_0*psi0+psi1'*m_rho_1*psi1)/2 # average fidelity

    a_theta = -(a_theta_0 + a_theta_1)/2
    a_kappa = -(a_kappa_0 + a_kappa_1)/2

    # Calculate gradients
    A = ma_rho_0 - S0.U*S0.Vt
    B = ma_rho_1 - S1.U*S1.Vt
    
    grad0 = -((A+A')*psi0 - 2*F0*psi0)/2/norm_0
    grad1 = -((B+B')*psi1 - 2*F1*psi1)/2/norm_1
    
    [F_avg,F], grad0, grad1, a_theta, a_kappa
end

function state_stabilization_fidelity(s::setting,psi0;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0: state to be stabilized

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)    
    rho_0 = psi0*psi0'

    function f(a,rho)
    end

    m_rho_0, ma_rho_0, a_theta, a_kappa = me_adjoint_solve(H, c_ops, rho_0, rho_0, s, theta, kappa,f)

    F = real(psi0'*m_rho_0*psi0)

    # Calculate gradients
    A = m_rho_0 + ma_rho_0    
    grad0 = ((A+A')*psi0 - 4*F*psi0)/2/norm_0
    
    # F, grad0/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, a_theta, a_kappa
end

function state_stabilization_trace_distance(s::setting,psi0;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0: state to be stabilized

    # get Hamiltonian and dissipators
    # H = sparse(sum(vcat([s.H0], [s.Hs[j]*theta_j for (j,theta_j) in enumerate(theta)])))
    H = copy(s.H0)
    for j in 1:length(theta)
        H += s.Hs[j]*theta[j]
    end
    # c_ops = vcat(s.A0, [s.As[i]*sqrt(kappa_i) for (i,kappa_i) in enumerate(kappa)])
    c_ops = copy(s.A0)
    for i in 1:length(kappa)
        push!(c_ops, s.As[i]*sqrt(kappa[i]))
    end

    norm_0, psi0 = normalize(psi0)    
    rho_0 = psi0*psi0'

    function f(a,rho_T)
        # here a must be a copy of the initial state rho_0
        S = svd(rho_T-a)
        mul!(a, S.U, S.Vt)
    end

    m_rho_0, ma_rho_0, a_theta, a_kappa = me_adjoint_solve(H, c_ops, rho_0, rho_0, s, theta, kappa,f)

    S = svd(m_rho_0 - rho_0)
    td = sum(S.S)

    F = real(psi0'*m_rho_0*psi0)

    # Calculate gradients
    A = ma_rho_0 - S.U*S.Vt 
    grad0 = -((A+A')*psi0 - 2*td*psi0)/2/norm_0
    
    [F,td/2], grad0, -a_theta, -a_kappa
end


function logical_qubit_average_fidelity_td(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    function f1(a,rho)
    end
    function f2(a,rho)
        tmp = trace(a', rho)
        tmp = tmp / abs(tmp)
        @. a *= tmp
    end

    results = pmap((rho,a,f) -> me_adjoint_solve_td(u, v, rho, a, s, theta, kappa,f), [rho_0, rho_1, rho_10], [rho_0+0.5*rho_1, rho_1+0.5*rho_0, rho_10],[f1,f1,f2])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]
    m_rho_10, ma_rho_10, a_theta_10, a_kappa_10 = results[3]

    ttt = psi1'*m_rho_10*psi0

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = psi1'*m_rho_0*psi1 + psi0'*m_rho_1*psi0 + 2*abs(ttt)

    F = real(F0/3 + F1/3 + F2/6) # average fidelity

    a_theta = (a_theta_0 + a_theta_1 + a_theta_10)/3
    a_kappa = (a_kappa_0 + a_kappa_1 + a_kappa_10)/3

    # Calculate gradients
    A = m_rho_0 + ma_rho_0*4.0/3.0 - ma_rho_1*2.0/3.0
    B = m_rho_1 + ma_rho_1*4.0/3.0 - ma_rho_0*2.0/3.0
    C = m_rho_10*ttt'/abs(ttt) + ma_rho_10
    
    g1 = (A+A')*psi0 - 4*F0*psi0
    g2 = (B+B')*psi0 + 2*conj(psi1'*C)[:] - 2*F2*psi0
    grad0 = (2*g1 + g2)/6/norm_0
    
    g1 = (B+B')*psi1 - 4*F1*psi1
    g2 = (A+A')*psi1 + 2*C*psi0 - 2*F2*psi1
    grad1 = (2*g1 + g2)/6/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end




















function pack_lambda(psi0,psi1,lambda)
    x = [real(psi0);imag(psi0)]
    append!(x, [real(psi1);imag(psi1)])
    append!(x, lambda)
    x
end

function unpack_lambda(x,psi0,psi1,lambda)
    dim = length(psi0)
    n = length(lambda)
    @. psi0 = x[1:dim] + 1.0im*x[dim+1:dim*2]
    @. psi1 = x[dim*2+1:dim*3] + 1.0im*x[dim*3+1:dim*4]
    @. lambda = x[dim*4+1:dim*4+n]
end

function learning_logical_qubit_new(s::setting,psi0,psi1,adam,niters;
                                lambda=Float64[],lambda_lb=Float64[],lambda_ub=Float64[],
                                f=grad_modifier_default,g=default_print,
                                grad_f=logical_qubit_average_fidelity)
    # reset the internal state of the adam optimizer
    ResetAdam(adam)

    # convert theta and kappa to angle variables to optimizer over
    alpha = similar(lambda)
    @. alpha = asin((2*lambda - lambda_ub - lambda_lb)/(lambda_ub - lambda_lb))

    # pre-allocate the gradients for alpha and beta
    grad_alpha = similar(lambda)

    x = pack_lambda(psi0, psi1, alpha)

    F_list = Float64[]

    for step in 1:niters
        F, grad_psi0, grad_psi1, grad_lambda = grad_f(s,psi0,psi1,lambda=lambda)
        push!(F_list,F)

        # print and save stuff (before the update step)
        g(step, F, psi0, psi1, lambda, grad_psi0, grad_psi1, grad_lambda, F_list)

        @. grad_alpha = (lambda_ub - lambda_lb)/2 * cos(alpha) * grad_lambda

        grad = pack_lambda(grad_psi0, grad_psi1, grad_alpha)
        f(grad) # modify the gradients in place; by default f is identity mapping
        GradAscentAdam(adam, x, grad, step)
        unpack_lambda(x, psi0, psi1, alpha)

        @. lambda = (lambda_ub + lambda_lb)/2 + (lambda_ub - lambda_lb)/2 * sin(alpha)

        # maintain orthogonality between psi0 and psi1
        r = (psi0'*psi1)/(psi0'*psi0)
        @. psi1 -= r * psi0
    end
    F_list
end

function logical_qubit_average_fidelity_td_new(s::setting,psi0,psi1;lambda=Float64[])
    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    function f1(a,rho)
    end
    function f2(a,rho)
        tmp = trace(a', rho)
        tmp = tmp / abs(tmp)
        @. a *= tmp
    end

    results = pmap((rho,a,f) -> me_adjoint_solve_td_new(rho, a, s, lambda, f), [rho_0, rho_1, rho_10], [rho_0+0.5*rho_1, rho_1+0.5*rho_0, rho_10],[f1,f1,f2])
    m_rho_0, ma_rho_0, a_lambda_0,  = results[1]
    m_rho_1, ma_rho_1, a_lambda_1,  = results[2]
    m_rho_10, ma_rho_10, a_lambda_10 = results[3]

    ttt = psi1'*m_rho_10*psi0

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = psi1'*m_rho_0*psi1 + psi0'*m_rho_1*psi0 + 2*abs(ttt)

    F = real(F0/3 + F1/3 + F2/6) # average fidelity

    a_lambda = (a_lambda_0 + a_lambda_1 + a_lambda_10)/3

    # Calculate gradients
    A = m_rho_0 + ma_rho_0*4.0/3.0 - ma_rho_1*2.0/3.0
    B = m_rho_1 + ma_rho_1*4.0/3.0 - ma_rho_0*2.0/3.0
    C = m_rho_10*ttt'/abs(ttt) + ma_rho_10
    
    g1 = (A+A')*psi0 - 4*F0*psi0
    g2 = (B+B')*psi0 + 2*conj(psi1'*C)[:] - 2*F2*psi0
    grad0 = (2*g1 + g2)/6/norm_0
    
    g1 = (B+B')*psi1 - 4*F1*psi1
    g2 = (A+A')*psi1 + 2*C*psi0 - 2*F2*psi1
    grad1 = (2*g1 + g2)/6/norm_1
    
    F, grad0, grad1, a_lambda
end




















function logical_qubit_entanglement_fidelity_td(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'
    rho_10 = psi1*psi0'

    function f1(a,rho)
    end
    function f2(a,rho)
        tmp = trace(a', rho)
        tmp = tmp / abs(tmp)
        @. a *= tmp
    end

    results = pmap((rho,a,f) -> me_adjoint_solve_td(u, v, rho, a, s, theta, kappa,f), [rho_0, rho_1, rho_10], [rho_0, rho_1, rho_10],[f1,f1,f2])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]
    m_rho_10, ma_rho_10, a_theta_10, a_kappa_10 = results[3]

    ttt = psi1'*m_rho_10*psi0

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1
    F2 = 2*abs(ttt)

    F = real(F0 + F1 + F2)/4 # entanglement fidelity

    a_theta = (a_theta_0 + a_theta_1 + 2*a_theta_10)/4
    a_kappa = (a_kappa_0 + a_kappa_1 + 2*a_kappa_10)/4

    # Calculate gradients
    A = m_rho_0 + ma_rho_0
    B = m_rho_1 + ma_rho_1
    C = m_rho_10*ttt'/abs(ttt) + ma_rho_10
    
    g1 = (A+A')*psi0 - 4*F0*psi0
    g2 = 2*conj(psi1'*C)[:] - 2*F2*psi0
    grad0 = (g1 + 2*g2)/4/norm_0
    
    g1 = (B+B')*psi1 - 4*F1*psi1
    g2 = 2*C*psi0 - 2*F2*psi1
    grad1 = (g1 + 2*g2)/4/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end

function biased_logical_qubit_average_fidelity_td(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'

    function f(a,rho)
    end

    results = pmap((rho,a,f) -> me_adjoint_solve_td(u, v, rho, a, s, theta, kappa,f), [rho_0, rho_1], [rho_0, rho_1],[f,f])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]

    F0 = psi0'*m_rho_0*psi0
    F1 = psi1'*m_rho_1*psi1

    F = real(F0+F1)/2 # average fidelity

    a_theta = (a_theta_0 + a_theta_1)/2
    a_kappa = (a_kappa_0 + a_kappa_1)/2

    # Calculate gradients
    A = m_rho_0 + ma_rho_0
    B = m_rho_1 + ma_rho_1
    
    grad0 = ((A+A')*psi0 - 4*F0*psi0)/2/norm_0
    grad1 = ((B+B')*psi1 - 4*F1*psi1)/2/norm_1
    
    # F, grad0/(1-F), grad1/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, grad1, a_theta, a_kappa
end

function biased_logical_qubit_trace_distance_td(s::setting,psi0,psi1;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0, psi1: current basis vectors of the logical subspace

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)
    norm_1, psi1 = normalize(psi1)
    
    rho_0 = psi0*psi0'
    rho_1 = psi1*psi1'

    function f(a,rho_T)
        # here a must be a copy of the initial state rho_0
        S = svd(rho_T-a)
        mul!(a, S.U, S.Vt)
    end

    results = pmap((rho,a,f) -> me_adjoint_solve_td(u, v, rho, a, s, theta, kappa,f), [rho_0, rho_1], [rho_0, rho_1],[f,f])
    m_rho_0, ma_rho_0, a_theta_0, a_kappa_0 = results[1]
    m_rho_1, ma_rho_1, a_theta_1, a_kappa_1 = results[2]

    S0 = svd(m_rho_0 - rho_0)
    S1 = svd(m_rho_1 - rho_1)

    F0 = sum(S0.S)
    F1 = sum(S1.S)

    F = real(F0+F1)/4 # average trace distance

    F_avg = real(psi0'*m_rho_0*psi0+psi1'*m_rho_1*psi1)/2 # average fidelity

    a_theta = -(a_theta_0 + a_theta_1)/2
    a_kappa = -(a_kappa_0 + a_kappa_1)/2

    # Calculate gradients
    A = ma_rho_0 - S0.U*S0.Vt
    B = ma_rho_1 - S1.U*S1.Vt
    
    grad0 = -((A+A')*psi0 - 2*F0*psi0)/2/norm_0
    grad1 = -((B+B')*psi1 - 2*F1*psi1)/2/norm_1
    
    [F_avg,F], grad0, grad1, a_theta, a_kappa
end

function state_stabilization_fidelity_td(s::setting,psi0;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0: state to be stabilized

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)    
    rho_0 = psi0*psi0'

    function f(a,rho)
    end

    m_rho_0, ma_rho_0, a_theta, a_kappa = me_adjoint_solve_td(u, v, rho_0, rho_0, s, theta, kappa,f)

    F = real(psi0'*m_rho_0*psi0)

    # Calculate gradients
    A = m_rho_0 + ma_rho_0    
    grad0 = ((A+A')*psi0 - 4*F*psi0)/2/norm_0
    
    # F, grad0/(1-F), a_theta/(1-F), a_kappa/(1-F)
    F, grad0, a_theta, a_kappa
end

function state_stabilization_trace_distance_td(s::setting,psi0;theta=Float64[],kappa=Float64[])
    # Adjoint method for learning the logical qubit subspace
    # Calculate the average fidelity and gradients w.r.t. psi0,psi1,theta,kappa
    # Parameters:
    #     - H0: constant part of the Hamiltonian
    #     - Hs: [H_j, j=1,2,...]
    #     - theta: [theta_j, j=1,2,...]
    #     - A0: list of the constant dissipators
    #     - As: [A_i, i==1,2,...]
    #     - kappa: [kappa_i, i=1,2,...]
    #     - psi0: state to be stabilized

    # get pulse shape for both Hs and As
    u = get_pulse(s.Mu,theta)
    v = get_pulse(s.Mv,kappa)

    norm_0, psi0 = normalize(psi0)    
    rho_0 = psi0*psi0'

    function f(a,rho_T)
        # here a must be a copy of the initial state rho_0
        S = svd(rho_T-a)
        mul!(a, S.U, S.Vt)
    end

    m_rho_0, ma_rho_0, a_theta, a_kappa = me_adjoint_solve_td(u, v, rho_0, rho_0, s, theta, kappa,f)

    S = svd(m_rho_0 - rho_0)
    td = sum(S.S)

    F = real(psi0'*m_rho_0*psi0)

    # Calculate gradients
    A = ma_rho_0 - S.U*S.Vt 
    grad0 = -((A+A')*psi0 - 2*td*psi0)/2/norm_0
    
    [F,td/2], grad0, -a_theta, -a_kappa
end

