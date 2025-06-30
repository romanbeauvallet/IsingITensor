#!usr/bin/env julia
module IsingITensor

include("Librairy.jl")

export hilberthalfspin, initnewmpshalfspin, initnewrandomhalfspin
export isingtensor, isinggates, tebdising, magnetization!, ising_magnetization, gates, isingtensorarray

end # module IsingITensor
