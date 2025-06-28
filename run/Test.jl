#!usr/bin/env julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using IsingITensor
using LinearAlgebra
using ITensors
using Plots
using ProgressMeter

######## Paramètres ###########

N = 20 #attention prendre N pair
s = 1 / 2
type = ["up"]
D = 10
Dmax = 10
Dmaxtebd = 100
J = 1
h = 0.5
dt = 1e-5
dim = 2
beta = 0.01
cutoff = 1e-15
n_sweep = 100
site_measure = div(N, 2)

shl = Index(dim, "horiz left")
shr = Index(dim, "horiz right")
svd = Index(dim, "vert down")
svu = Index(dim, "vert up")

####### Test ###########

randomps = initnewrandomhalfspin(N, Dmax)
mps = deepcopy(randomps)
#@show typeof(mps)
#tensor = isinggates(randomps, beta, J, "even")
#@show length(tensor), tensor[4]

magnet = magnetization!(update, beta, site_measure, J, Dmaxtebd, cutoff)

Betalist = collect(0.01:0.01:1)

####### Data #########################

Mpslist = Vector{}()
Magnetlist = Vector{}()
Magnetexact = Vector{}()

function void()
    @showprogress for i in eachindex(Betalist)
        update = tebdising(mps, Betalist[i], J, cutoff, n_sweep, Dmaxtebd)
        push!(Mpslist, update)
        magnet = magnetization!(update, Betalist[i], site_measure, J, Dmaxtebd, cutoff)
        push!(Magnetlist, magnet)
        push!(Magnetexact, ising_magnetization(Betalist[i]))
    end
end

void()
############### Graphs ############

gr()

plot(Betalist, Magnetlist, label="tebd")
plot!(Betalist, Magnetexact, label="exact")