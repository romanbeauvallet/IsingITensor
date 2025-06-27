#!usr/bin/env julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using IsingITensor
using LinearAlgebra
using ITensors

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
beta = 1
cutoff = 1e-15

shl = Index(dim, "horiz left")
shr = Index(dim, "horiz right")
svd = Index(dim, "vert down")
svu = Index(dim, "vert up")

####### Test ###########

randomps = initnewrandomhalfspin(N, Dmax)
mps = deepcopy(randomps)
tensor = isinggates(randomps, beta, shl => shr, svd => svu, J)
#@show length(tensor), tensor[4]

tebdorder2(mps, tensor, cutoff, Dmaxtebd)

