#!usr/bin/env julia
module IsingITensor

include("Librairy.jl")

export hilberthalfspin, initnewmpshalfspin, initnewrandomhalfspin
export isingtensor, isinggates, tebdising, magnetization!, ising_magnetization, gates, isingtensorarray, tebdising2, magnetization2!
import ITensors: has_tag

end # module IsingITensor
