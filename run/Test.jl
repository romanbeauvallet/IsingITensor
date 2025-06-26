#!usr/bin/env julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using IsingITensor
using LinearAlgebra

######## Paramètres ###########

N = 20
s = 1 / 2
type = ["up"]
D = 10
Dmax = 10
J = 1
h = 0.5
dt = 1e-5

####### Test ###########

randomps = initnewrandomhalfspin(N, Dmax)
mps = initnewmpshalfspin(N, D)
@show length(randomps)