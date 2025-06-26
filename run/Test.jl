#!usr/bin/env julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using IsingITensor

######## Paramètres ###########

N = 20 
s = 1/2 
type = ["up"]

####### Test ###########

@show size.(initnewmps(N, s, type))