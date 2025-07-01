#!usr/bin/env julia
using ITensors
using ITensorMPS
using LinearAlgebra
using QuadGK
using TensorOperations

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
return ising tensor with array type 
"""
function isingtensorarray(beta, J, sz::Bool=false)
    T = zeros(2, 2, 2, 2)
    T[1, 1, 1, 1] = 1
    T[2, 2, 2, 2] = 1
    if sz == true
        T[2, 2, 2, 2] = -1
    end
    Q = [exp(beta * J) exp(-beta * J); exp(-beta * J) exp(beta * J)]
    X = sqrt(Q)
    @tensor D[i, j, k, l] := T[a, b, c, d] * X[i, a] * X[j, b] * X[k, c] * X[l, d]
    return D
end

"""
return the list of gates 
"""
function gates(mps, beta, J, parity::String, sz::Bool=false)
    tensor = isingtensorarray(beta, J, sz)
    sites = siteinds(mps)
    operator = Vector{}()
    if parity == "even"
        for i in 1:1:div(length(mps), 2)
            push!(operator, op(tensor, siteind(mps, 2 * i - 1), siteind(mps, 2 * i)))
        end
    elseif parity == "odd"
        for i in 1:1:div(length(mps), 2)-1
            push!(operator, op(tensor, siteind(mps, 2 * i), siteind(mps, 2 * i + 1)))
        end
    end
    return operator
end


""" 
return the vector of ising gates to apply on the MPS
"""
function isinggates(mps, beta, J, parity::String, sz::Bool, h=0) #essayer avec op et le array pour isingtensor
    n = length(mps)
    q = div(n, 2) #a gate applies on two sites
    sₕ, sₕ′ = (Index(2, "horiz left"), Index(2, "horiz right"))
    sᵥ, sᵥ′ = (Index(2, "vert down"), Index(2, "vert up"))
    @assert dim(sₕ) == dim(sᵥ)
    d = dim(sₕ)
    T = ITensor(sₕ, sₕ′, sᵥ, sᵥ′)
    for i in 1:d
        T[i, i, i, i] = 1.0
    end
    if sz == true
        T[2, 2, 2, 2] = -1.0
    end
    s̃ₕ, s̃ₕ′, s̃ᵥ, s̃ᵥ′ = sim.((sₕ, sₕ′, sᵥ, sᵥ′))
    T̃ = T * δ(sₕ, s̃ₕ) * δ(sₕ′, s̃ₕ′) * δ(sᵥ, s̃ᵥ) * δ(sᵥ′, s̃ᵥ′)
    Q = [exp(beta * J) exp(-beta * J); exp(-beta * J) exp(beta * J)]
    X = sqrt(Q)
    Xₕ = itensor(vec(X), s̃ₕ, sₕ)
    Xₕ′ = itensor(vec(X), s̃ₕ′, sₕ′)
    Xᵥ = itensor(vec(X), s̃ᵥ, sᵥ)
    Xᵥ′ = itensor(vec(X), s̃ᵥ′, sᵥ′)
    inter = T̃ * Xₕ′ * Xᵥ′ * Xₕ * Xᵥ
    inds_inter = inds(inter)
    if parity == "even"
        gateslist = Vector{ITensor}(undef, q)
        for i in 1:1:q
            s1 = siteind(mps, 2 * i - 1)    # indice physique du site i
            s2 = siteind(mps, 2 * i)  # indice physique du site i+1
            # On crée deux nouveaux indices "primés" (output)
            s1p = prime(s1)
            s2p = prime(s2)
            inter_aligned = replaceinds(inter, (inds_inter[1] => s1p, inds_inter[2] => s1, inds_inter[3] => s2, inds_inter[4] => s2p))
            #@show inds(inter_aligned)
            gateslist[i] = inter_aligned
        end
    elseif parity == "odd"
        gateslist = Vector{ITensor}(undef, q - 1)
        for i in 1:1:q-1
            s1 = siteind(mps, 2 * i)  # indice physique du site i
            s2 = siteind(mps, 2 * i + 1)   # indice physique du site i+1
            # On crée deux nouveaux indices "primés" (output)
            s1p = prime(s1)
            s2p = prime(s2)
            #@show s1, s1ps
            #@show inds_inter[1], inds_inter[2]
            inter_aligned = replaceinds(inter, (inds_inter[1] => s1p, inds_inter[2] => s1, inds_inter[3] => s2, inds_inter[4] => s2p))
            #@show inds(inter_aligned)
            gateslist[i] = inter_aligned
        end
    end
    return gateslist
end

#utiliser la fonction ITensorMPS.TwoSiteGate

"""
mps -- boundary mps 
gatelist -- vector of gates you apply on mps

return the converged mps for the contraction of the 2D Ising tensor networks with boundary mps algorithm
"""
function tebdising(mps, beta, J, cutoff, n_sweep, Dmaxtebd)
    n = length(mps)
    copymps = deepcopy(mps)
    #@show length(gatelist)
    for j in 1:n_sweep
        #@show j
        #@show gatelist[1]
        #@show copymps[1], copymps[2]
        #@show j, copymps
        gatelist1 = isinggates(mps, beta, J, "even", false)
        #@show length(gatelist1)
        copymps = apply(gatelist1, copymps; maxdim=Dmaxtebd, cutoff=cutoff)
        normalize!(copymps)
        #@show copymps
        gatelist2 = isinggates(copymps, beta, J, "odd", false)
        #@show length(gatelist2)
        copymps = apply(gatelist2, copymps; maxdim=Dmaxtebd, cutoff=cutoff)
        normalize!(copymps)
    end
    return copymps
end
"""

return one tebd sweep on the mps with the gates function rather than tebdising
"""
function tebdising2(mps, beta, J, cutoff, n_sweep, Dmaxtebd)
    n = length(mps)
    copymps = deepcopy(mps)
    #@show length(gatelist)
    for j in 1:n_sweep
        #@show j
        #@show gatelist[1]
        #@show copymps[1], copymps[2]
        #@show j, copymps
        gatelist1 = gates(mps, beta, J, "even", false)
        #@show length(gatelist1)
        for ope in gatelist1
            copymps = apply(ope, copymps; maxdim=Dmaxtebd, cutoff=cutoff)
        end
        normalize!(copymps)
        #@show copymps
        gatelist2 = gates(copymps, beta, J, "odd", false)
        for ope in gatelist2
            copymps = apply(ope, copymps; maxdim=Dmaxtebd, cutoff=cutoff)

        end
        #@show length(gatelist2)
        normalize!(copymps)
    end
    return copymps
end


"""
return the magnetization of the site i 
"""
function magnetization!(mps, beta, i, J, Dmaxtebd, cutoff)
    #@show typeof(mps)
    ind_env = [siteind(mps, i), siteind(mps, i + 1)]
    orthogonalize!(mps, i)
    #@show length(mps)
    #ne pas faire apply mais juste produit * avec des type ITensor et des bons index 
    subsites = siteinds(mps)[i:i+1]
    env = MPS(subsites)
    env[1] = mps[i]
    env[2] = mps[i+1]
    #@show typeof(env)
    site_norm = isinggates(env, beta, J, "even", false)[1]
    ind_site_norm = inds(site_norm)
    site_norm_index = replaceinds(site_norm, (ind_site_norm[1] => prime(ind_env[1]), ind_site_norm[2] => ind_env[1], ind_site_norm[3] => ind_env[2], ind_site_norm[4] => prime(ind_env[2])))
    site_meas = isinggates(env, beta, J, "even", true)[1]
    ind_site_meas = inds(site_meas)
    site_meas_index = replaceinds(site_meas, (ind_site_meas[1] => prime(ind_env[1]), ind_site_meas[2] => ind_env[1], ind_site_meas[3] => ind_env[2], ind_site_meas[4] => prime(ind_env[2])))
    #@show env, site_meas_index
    env_norm = apply(site_norm_index, env; maxdim=Dmaxtebd, cutoff=cutoff)
    env_meas = apply(site_meas_index, env; maxdim=Dmaxtebd, cutoff=cutoff)
    m = inner(env_meas, env_meas)
    n = inner(env_norm, env_norm)
    return m / n
end

function magnetization2!(mps, beta, i, J, Dmaxtebd, cutoff)
    orthogonalize!(mps, i)
    env_init = mps[i:i+1]
    @show env_init
    env = env_init[1] * env_init[2]
    tensorising = isingtensorarray(beta, J, true) 
    index  = filter(i -> hastags(i, "Site"), inds(env))
    @show inds(env)
    gatemagnet = op(tensorising, index[1], index[2])
    semicontract = gatemagnet * env
    index_semicontract  = filter(i -> hastags(i, "Site"), inds(semicontract))
    dag = conj(env)
    index_dag  = filter(i -> hastags(i, "Site"), inds(env))
    replaceinds!(dag, (index_dag[1] => index_semicontract[1], index_dag[2] => index_semicontract[2]))
    m = inner(semicontract, dag)[]
    tensorising2 = isingtensorarray(beta, J, false)
    gatemagnet2 = op(tensorising2, index[1], index[2])
    semicontract2 = gatemagnet2 * env
    index_semicontract2  = filter(i -> hastags(i, "Site"), inds(semicontract2))
    dag2 = conj(env)
    index_dag2  = filter(i -> hastags(i, "Site"), inds(env))
    replaceinds!(dag2, (index_dag2[1] => index_semicontract2[1], index_dag2[2] => index_semicontract2[2]))
    n = inner(semicontract2, dag2)[]
    return m/n
end