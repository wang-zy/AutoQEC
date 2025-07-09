mutable struct Adam
	beta1::Float64
    beta2::Float64
    alpha::Float64
    eps::Float64
    du::Array{Float64,1}
    dm::Array{Float64,1}
    dv::Array{Float64,1}
    function Adam(;lr=0.001, dim=10, beta1=0.9, beta2=0.999, eps=1e-8)
    	du = zeros(dim)
    	dm = zeros(dim)
    	dv = zeros(dim)
    	new(beta1, beta2, lr, eps, du, dm, dv)
    end
end

function GradAscentAdam(adam, x, grad, step)
	adam.dm = adam.beta1*adam.dm+(1-adam.beta1)*grad
    adam.dv = adam.beta2*adam.dv+(1-adam.beta2)*(grad.^2)
    dmhat = adam.dm/(1-adam.beta1.^step)
    dvhat = adam.dv/(1-adam.beta2.^step)
    @. x += adam.alpha * dmhat / (sqrt(dvhat) + adam.eps)
end

function ResetAdam(adam)
	@. adam.du *= 0
	@. adam.dm *= 0
	@. adam.dv *= 0
end

function pack(psi0,psi1,theta,kappa)
	x = [real(psi0);imag(psi0)]
	append!(x, [real(psi1);imag(psi1)])
	append!(x, theta)
	append!(x, kappa)
	x
end

function pack(psi0,theta,kappa)
    x = [real(psi0);imag(psi0)]
    append!(x, theta)
    append!(x, kappa)
    x
end

function unpack(x,psi0,psi1,theta,kappa)
	dim = length(psi0)
	n1 = length(theta)
	n2 = length(kappa)
	@. psi0 = x[1:dim] + 1.0im*x[dim+1:dim*2]
	@. psi1 = x[dim*2+1:dim*3] + 1.0im*x[dim*3+1:dim*4]
	@. theta = x[dim*4+1:dim*4+n1]
	@. kappa = x[dim*4+n1+1:dim*4+n1+n2]
end

function unpack(x,psi0,theta,kappa)
    dim = length(psi0)
    n1 = length(theta)
    n2 = length(kappa)
    @. psi0 = x[1:dim] + 1.0im*x[dim+1:dim*2]
    @. theta = x[dim*2+1:dim*2+n1]
    @. kappa = x[dim*2+n1+1:dim*2+n1+n2]
end
