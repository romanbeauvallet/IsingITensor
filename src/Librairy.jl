#!usr/bin/env julia
using ITensors
using ITensorMPS
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

return the right hilbert space for the system
"""
function hilberthalfspin(N)
    physical = ITensors.siteinds("SpinHalf", N)
    return physical
end

"""
N -- number of sites
D -- bond dimension in the MPS (applied to each tensor)

return an initialization of a MPS, note that the intrication can be nul with D = 1
"""
function initnewmpshalfspin(N, D)
    site = hilberthalfspin(N)
    psi = MPS(site, linkdims=D)
    return psi
end

"""
N -- number of sites
Dmax -- maximum bond dimension in the MPS 

return an random initialization of a MPS, the MPS is already normalized 
"""
function initnewrandomhalfspin(N, Dmax)
    site = hilberthalfspin(N)
    psi = random_mps(site, linkdims=Dmax)
    return psi
end

"""
beta -- inverse temmperature
J -- coupling constant

return the Ising tensor used to converge the MPS to its staidy state at the temperature beta without MPO
"""
function isingtensor(J, i, h, t)
    ZZ = op("Z", sites, i) * op("Z", sites, i + 1)
    X = op("X", sites, i)
    G = exp(-t * J * ZZ - t * h * X)
    return replaceprime(G, 0 => 1)
end

"""
return the ising tensor for tebd with MPO format but without disorder
"""
function isingtensormpo(beta, j, dim)
    sₕ = Index(dim, "horiz")
    sₕ′ = Index(dim, "horiz'")
    sᵥ = Index(dim, "vert")
    sᵥ′ = Index(dim, "vert'")
    MPO = ising_mpo(sₕ => sₕ′, sᵥ => sᵥ′, beta, J=j)
    return MPO
end