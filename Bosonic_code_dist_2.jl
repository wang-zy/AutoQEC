using JLD
using Distributed
addprocs(3)

@everywhere push!(LOAD_PATH, pwd()*"/../../AutoQEC/src") # modify the path correspondingly so that Julia could find the path for AutoQEC.jl
@everywhere using AutoQEC
using Random

seed = rand(1:100000,1)[1]
# seed = 9602 # if you want to reproduce the result in the google doc
Random.seed!(seed)
println(seed)


N = 20
Nq = 2
a = kron(destroy(N),id(Nq))
aq = kron(id(N),destroy(Nq))

H0 = 0.0*a'*a

Hs=Array{CP,2}[]
for i = 1:N-1
	push!(Hs, kron(basis(N,i-1)*basis(N,i)',id(Nq))*aq+kron(basis(N,i)*basis(N,i-1)',id(Nq))*aq')
	push!(Hs, 1im*(kron(basis(N,i-1)*basis(N,i)',id(Nq))*aq-kron(basis(N,i)*basis(N,i-1)',id(Nq))*aq'))
end
for i = 1:N-1
	push!(Hs, kron(basis(N,i-1)*basis(N,i)',id(Nq))*aq'+kron(basis(N,i)*basis(N,i-1)',id(Nq))*aq)
	push!(Hs, 1im*(kron(basis(N,i-1)*basis(N,i)',id(Nq))*aq'-kron(basis(N,i)*basis(N,i-1)',id(Nq))*aq))
end
for i = 1:N-2
	push!(Hs, kron(basis(N,i-1)*basis(N,i+1)',id(Nq))*aq+kron(basis(N,i+1)*basis(N,i-1)',id(Nq))*aq')
	push!(Hs, 1im*(kron(basis(N,i-1)*basis(N,i+1)',id(Nq))*aq-kron(basis(N,i+1)*basis(N,i-1)',id(Nq))*aq'))
end
for i = 1:N-2
	push!(Hs, kron(basis(N,i-1)*basis(N,i+1)',id(Nq))*aq'+kron(basis(N,i+1)*basis(N,i-1)',id(Nq))*aq)
	push!(Hs, 1im*(kron(basis(N,i-1)*basis(N,i+1)',id(Nq))*aq'-kron(basis(N,i+1)*basis(N,i-1)',id(Nq))*aq))
end

gamma = 2*pi*0.1
gamma_q = 2*pi*20
A0 = [sqrt(gamma)*a, sqrt(gamma_q)*aq]

As = Array{CP,2}[]

psi0 = randn(N) + 1.0im * randn(N)
psi1 = randn(N) + 1.0im * randn(N)
r = (psi0'*psi1)/(psi0'*psi0)
@. psi1 -= r * psi0
norm_0 = sum(real(psi0).^2) + sum(imag(psi0).^2)
norm_1 = sum(real(psi1).^2) + sum(imag(psi1).^2)
psi0 = kron(psi0/sqrt(norm_0), basis(Nq,0))
psi1 = kron(psi1/sqrt(norm_1), basis(Nq,0))

n_Hs = length(Hs)
n_As = length(As)

theta_lb = -2*pi*ones(n_Hs)*10
theta_ub = 2*pi*ones(n_Hs)*10
theta = (theta_lb + (theta_ub - theta_lb) .* rand(n_Hs)) * 0.01

kappa_lb = Float64[]
kappa_ub = Float64[]
kappa = Float64[]

fname = "data/dist_2/data_"*string(seed)*"_"
function g(step, F_final, psi0, psi1, theta, args...)
    if step%1 == 0
        a0 = (psi0'*a*psi0)[1]/sum(abs.(psi0).^2)
        a1 = (psi1'*a*psi1)[1]/sum(abs.(psi1).^2)
        println(step,": ",F_final, "    alpha:",(a0+a1)/2)
        if step%100 == 0
            save(fname*string(F_final)*".jld", "psi0", psi0, "psi1", psi1, "theta", theta/2/pi, "kappa", kappa/2/pi, "F_final", F_final)
        end
    end
end

const s = setting(H0=H0,Hs=Hs,A0=A0,As=As,tmax=0.5,nsteps=1000,t=1)

kappa_min = minimum([gamma, gamma_q])
println("Single qubit fidelity:", (exp(-kappa_min*s.tmax)+2*exp(-kappa_min*s.tmax/2)+3)/6)

adam = Adam(lr=0.001, dim=N*Nq*4+n_Hs+n_As)

grad_f = logical_qubit_average_fidelity
learning_logical_qubit(s,psi0,psi1,adam,50000,theta=theta,theta_lb=theta_lb,theta_ub=theta_ub,kappa=kappa,kappa_lb=kappa_lb,kappa_ub=kappa_ub,g=g,grad_f=grad_f)

