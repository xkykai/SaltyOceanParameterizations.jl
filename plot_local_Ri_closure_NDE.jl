using JLD2
using Lux
using CairoMakie
using Optimization
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using SeawaterPolynomials.TEOS10
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations
using SaltyOceanParameterizations: calculate_Ri
using SciMLBase

FILE_DIR = "./training_output/local_diffusivity_NDE_gradient_relu_noclamp"

filename = "$(FILE_DIR)/training_results_2.jld2"

file = jldopen(filename, "r")

NN = file["NN"]
st = file["st_NN"]

res = file["res"]

NN([0], res.u, st)

Ris = -50:0.01:50
νs = zeros(length(Ris))
κs = zeros(length(Ris))

for (i, Ri) in enumerate(Ris)
    νs[i], κs[i] = first(NN([Ri], res.u, st)) ./ 10 .+ 1e-5
end

#%%
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel="Ri", ylabel="Diffusivities (m² s⁻¹)")
lines!(ax, Ris, νs, label="ν")
lines!(ax, Ris, κs, label="κ")
axislegend(ax, position=:rt)
display(fig)
save("$(FILE_DIR)/Ri_sweep.png", fig)
#%%