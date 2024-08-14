using SaltyOceanParameterizations: find_min, find_max
using JLD2
using CairoMakie
using Printf
using Colors
using Oceananigans

LES_FILE_DIRS = [
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_dataset = [FieldDataset(FILE_DIR) for FILE_DIR in LES_FILE_DIRS]

Qᵁs = [0.0, -0.0002, -0.0005]
Qᵀ = 0.0003
Qˢ = -3.0e-5
f₀ = 8.0e-5

#%%
fig = Figure(size=(2000, 600))
axu = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="z")
axv = CairoMakie.Axis(fig[1, 2], xlabel="v", ylabel="z")
axT = CairoMakie.Axis(fig[1, 3], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 4], xlabel="S", ylabel="z")
axρ = CairoMakie.Axis(fig[1, 5], xlabel="ρ", ylabel="z")

Nz = field_dataset[1]["ubar"].grid.Nz
Nt = length(field_dataset[1]["ubar"].times)
zC = field_dataset[1]["ubar"].grid.zᵃᵃᶜ[1:Nz]

ts = field_dataset[1]["ubar"].times

n = Observable(1)

u_truthₙ = [@lift interior(data["ubar"][$n], 1, 1, :) for data in field_dataset]
v_truthₙ = [@lift interior(data["vbar"][$n], 1, 1, :) for data in field_dataset]
T_truthₙ = [@lift interior(data["Tbar"][$n], 1, 1, :) for data in field_dataset]
S_truthₙ = [@lift interior(data["Sbar"][$n], 1, 1, :) for data in field_dataset]
ρ_truthₙ = [@lift interior(data["ρbar"][$n], 1, 1, :) for data in field_dataset]

ulim = (find_min([data["ubar"] for data in field_dataset]..., -1e-8), find_max([data["ubar"] for data in field_dataset]..., 1e-8))
vlim = (find_min([data["vbar"] for data in field_dataset]..., -1e-8), find_max([data["vbar"] for data in field_dataset]..., 1e-8))

labels = ["Qᵘ = $(Qᵁs[i]) m² s⁻²" for i in 1:length(Qᵁs)]

for (i, label) in enumerate(labels)
    lines!(axu, u_truthₙ[i], zC, label=label)
    lines!(axv, v_truthₙ[i], zC, label=label)
    lines!(axT, T_truthₙ[i], zC, label=label)
    lines!(axS, S_truthₙ[i], zC, label=label)
    lines!(axρ, ρ_truthₙ[i], zC, label=label)
end

axislegend(axu, position=:lb)

xlims!(axu, ulim)
xlims!(axv, vlim)

title = @lift "Varying Wind Stress, Qᵀ = $(Qᵀ) °C m s⁻¹, Qˢ = $(Qˢ) psu m s⁻¹, f = $(f₀) s⁻¹, $(round(ts[$n] / 24 / 60^2, digits=2)) days"

Label(fig[0, :], title, font=:bold, tellwidth=false)

display(fig)

CairoMakie.record(fig, "./Data/LES_shear_QT_$(Qᵀ)_QS_$(Qˢ)_f_$(f₀).mp4", 1:Nt, framerate=20) do nn
    n[] = nn
end
#%%