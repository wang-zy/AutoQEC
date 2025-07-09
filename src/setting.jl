const CP = Complex{Float64} # complex type

mutable struct setting
	H0::SparseMatrixCSC{CP,Int64}
	Hs_fast::Array{SparseMatrixCSC{CP,Int64},1}
    Hs::Array{SparseMatrixCSC{CP,Int64},1}
	A0::Array{SparseMatrixCSC{CP,Int64},1}
	A0_dag::Array{SparseMatrixCSC{CP,Int64},1}
	A0_dag_A0::SparseMatrixCSC{CP,Int64}
	As::Array{SparseMatrixCSC{CP,Int64},1}
	As_dag::Array{SparseMatrixCSC{CP,Int64},1}
	As_dag_As_fast::Array{SparseMatrixCSC{CP,Int64},1}
    As_dag_As::Array{SparseMatrixCSC{CP,Int64},1}

	tmax::Float64
    nsteps::Int64
    h::Float64
    # revolve algorithm related
    ncpts::Int64                    # number of checkpoints
    seq::Array{Array{Int64,1},1}    # sequence of steps

    # transformation matrix for the basis functions
    Mu::Array{Array{Float64,2},1}      # list of array with length n_Hs, array shape [n_us, nsteps+1]
    Mv::Array{Array{Float64,2},1}      # list of array with length n_As, array shape [n_vs, nsteps+1]

    # td
    uus
    vvs
    us_grad
    vs_grad
    u::Array{Float64,1}
    v::Array{Float64,1}

    function setting(;H0::Array{CP,2}=reshape([0.0+0.0im], (1,1)), Hs=Array{CP,2}[], 
                     A0=Array{CP,2}[], As=Array{CP,2}[], 
                     us=Array{Function,1}[], vs=Array{Function,1}[], 
                     uus=nothing, vvs=nothing, us_grad=nothing, vs_grad=nothing,
                     tmax=1.0, nsteps=1000, t=1)
    	# check if Hamiltonians are Hermitian
        if sum(abs.(H0'-H0))>1e-16
            error("Hamiltonians must be Hermitian!")
        end
        for H in Hs
            if sum(abs.(H'-H))>1e-16
                error("Hamiltonians must be Hermitian!")
            end
        end

        dim = size(H0)[1]
    	if size(H0) == (1,1)
    		if length(Hs) > 0
    			dim = size(Hs[1])[1]
    		else
    			if length(A0) > 0
    				dim = size(A0[1])[1]
    			else
    				if length(As) > 0
    					dim = size(As[1])[1]
    				else
    					error("Invalid inputs!")
    				end
    			end
    		end
    		H0_sp = sparse(zeros(CP, dim, dim))
    	else
    		H0_sp = sparse(-1.0im*H0)
    	end
    	Hs_sp = [sparse(-1.0im*H) for H in Hs]

        # common sparsity pattern
        H0_sp, Hs_sp_2 = sparse_H(H0_sp, Hs_sp)

    	A0_sp = [sparse(A) for A in A0]
    	A0_dag_sp = [sparse(A') for A in A0]
        A0_dag_A0 = sum([A'*A/2 for A in A0])

    	As_sp = [sparse(A) for A in As]
    	As_dag_sp = [sparse(A') for A in As]
    	As_dag_As_sp_fast = [sparse(A'*A/2) for A in As]

        # common sparsity pattern
        A0_dag_A0_sp, As_dag_As_sp = sparse_H(A0_dag_A0, As_dag_As_sp_fast)

        if length(us)+length(vs)>0
            if length(Hs) != length(us) || length(As) != length(vs)
                error("Invalid inputs!")
            end

            # time dependent case
            # double nsteps for controling ODE truncation error when calculating adjoint gradients
            # for a_theta and a_kappa with RK4
            nsteps *= 2

            # the basis functions have to be further interpolated by a factor of 2 to control
            # ODE error in forward and backward time evolution
            Mu = [basisMatrix(us[i], tmax, 2*nsteps) for i = 1:length(us)] # size: [length(us),2*nsteps+1]
            Mv = [basisMatrix(vs[i], tmax, 2*nsteps) for i = 1:length(vs)] # size: [length(vs),2*nsteps+1]
        else
            Mu = Array{Float64,2}[]
            Mv = Array{Float64,2}[]
        end

        if uus != nothing || vvs != nothing
            nsteps *= 2
        end
        h = tmax/nsteps
        ncpts, seq = revolve_schedule(nsteps, t)

    	new(H0_sp, Hs_sp, Hs_sp_2, A0_sp, A0_dag_sp, A0_dag_A0_sp, 
            As_sp, As_dag_sp, As_dag_As_sp_fast, As_dag_As_sp, 
            tmax, nsteps, h, ncpts, seq, Mu, Mv, uus, vvs, us_grad, vs_grad, zeros(length(Hs)), zeros(length(As)))
    end
end
