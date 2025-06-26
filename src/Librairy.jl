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
function evolutiontensor(J, i, h, dt)
    gatelist = Vector{}
    for i in 1:N-1
        ZZ = op("Z", sites, i) * op("Z", sites, i + 1)
        X = op("X", sites, i)
        G = exp(-dt * J * ZZ - dt * h * X)
        push!(gatelist, replaceprime(G, 0 => 1))
    end
end

"""
beta -- inverse temperature
j -- coupling constant
dim -- physical dimension

return the ising tensor for tebd with MPO format but without disorder
"""
function isingtensormpo(beta, j, dim=2)
    sₕ = Index(dim, "horiz")
    sₕ′ = Index(dim, "horiz'")
    sᵥ = Index(dim, "vert")
    sᵥ′ = Index(dim, "vert'")
    MPO = ising_mpo(sₕ => sₕ′, sᵥ => sᵥ′, beta, J=j)
    return MPO
end

"""

return the vector of ising gates to apply on the MPS
"""
function isinggates(mps, beta, J, h=0)
    gateslist = Vector{}
    n = length(mps)
    for i in 1:n-1
        sₕ, sₕ′ = pair_sₕ
        sᵥ, sᵥ′ = pair_sᵥ
        @assert dim(sₕ) == dim(sᵥ)
        d = dim(sₕ)
        T = ITensor(sₕ, sₕ′, sᵥ, sᵥ′)
        for i in 1:d
            T[i, i, i, i] = 1.0
        end
        s̃ₕ, s̃ₕ′, s̃ᵥ, s̃ᵥ′ = sim.((sₕ, sₕ′, sᵥ, sᵥ′))
        T̃ = T * δ(sₕ, s̃ₕ) * δ(sₕ′, s̃ₕ′) * δ(sᵥ, s̃ᵥ) * δ(sᵥ′, s̃ᵥ′)
        Q = [exp(beta * J) exp(-beta * J); exp(-beta * J) exp(beta * J)]
        X = sqrt(Q)
        Xₕ = itensor(vec(X), s̃ₕ, sₕ)
        Xₕ′ = itensor(vec(X), s̃ₕ′, sₕ′)
        Xᵥ = itensor(vec(X), s̃ᵥ, sᵥ)
        Xᵥ′ = itensor(vec(X), s̃ᵥ′, sᵥ′)
        push!(gateslist, T̃ * Xₕ′ * Xᵥ′ * Xₕ * Xᵥ)
    end
end

"""

"""
function tebdorder2(mps, gatelist)
    
end