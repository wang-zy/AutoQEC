module AutoQEC

using Distributed
using LinearAlgebra
using SparseArrays


include("setting.jl")
include("util.jl")
include("adjoint.jl")
include("adjoint_gradient.jl")
include("evolve.jl")
include("optimizer.jl")
include("revolve.jl")
include("quantum.jl")
include("evaluate.jl")


export CP, setting, id, destroy, basis, coherent, displace, tensor, wigner, Adam
export sigmax, sigmay, sigmaz, sigmam, sigmap
export learning_logical_qubit, learning_state_stabilization, learning_state_stabilization_map

# export gradients function for different loss functions
export logical_qubit_average_fidelity
export logical_qubit_entanglement_fidelity
export biased_logical_qubit_average_fidelity
export biased_logical_qubit_trace_distance
export state_stabilization_fidelity
export state_stabilization_trace_distance
export logical_qubit_average_fidelity_td
export logical_qubit_entanglement_fidelity_td
export biased_logical_qubit_average_fidelity_td
export biased_logical_qubit_trace_distance_td
export state_stabilization_fidelity_td
export state_stabilization_trace_distance_td

export learning_logical_qubit_new
export logical_qubit_average_fidelity_td_new

export me_solve, me_solve_td



end