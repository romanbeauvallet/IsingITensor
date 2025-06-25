#!usr/bin/env julia
using ITensors
using LinearAlgebra
using QuadGK

############## Exact Result ############

const βc = 0.5 * log(√2 + 1)

function ising_free_energy(β::Real, J::Real=1.0)
    k = β * J
    c = cosh(2 * k)
    s = sinh(2 * k)
    xmin = 0.0
    xmax = π
    integrand(x) = log(c^2 + √(s^4 + 1 - 2 * s^2 * cos(x)))
    integral, err = quadgk(integrand, xmin, xmax)::Tuple{Float64,Float64}
    return -(log(2) + integral / π) / (2 * β)
end

function ising_magnetization(β::Real)
    β > βc && return (1 - sinh(2 * β)^(-4))^(1 / 8)
    return 0.0
end

############## Functions #################

"""
N -- number of sites
s -- spin value

return the right hilbert space for the system
"""
function hilbert(N, s)
    return ITensors.siteinds("S = $s", N)
end

"""
N -- number of sites
s -- spin value
type -- specify the value of the physical indexes

return an initialization of a MPS
"""
function initnewmps(N, s, type::Vector{})
    site = hilbert(N, s)
    psi = ITensor.MPS(site, type)
end


