using Oceananigans
using JLD2
using CairoMakie

filenames_2D = (
    SL32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_1.0_64.0_Nxz_1_32",
    SL64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_1.0_64.0_Nxz_1_64",
    SL128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_1.0_64.0_Nxz_1_128",

    AMD32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_1.0_64.0_Nxz_1_32",
    AMD64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_1.0_64.0_Nxz_1_64",
    AMD128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_1.0_64.0_Nxz_1_128",

    W032 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_1.0_64.0_Nxz_1_32",
    W064 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_1.0_64.0_Nxz_1_64",
    W0128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_1.0_64.0_Nxz_1_128",

    W532 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_1.0_64.0_Nxz_1_32",
    W564 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_1.0_64.0_Nxz_1_64",
    W5128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_1.0_64.0_Nxz_1_128",

    WA32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_1.0_64.0_Nxz_1_32",
    WA64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_1.0_64.0_Nxz_1_64",
    WA128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_1.0_64.0_Nxz_1_128"
)

filenames_3D = (
    SL32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_128.0_64.0_Nxz_64_32",
    SL64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_128.0_64.0_Nxz_128_64",
    SL128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_SmagorinskyLilly_Lxz_128.0_64.0_Nxz_256_128",

    AMD32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_128.0_64.0_Nxz_64_32",
    AMD64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_128.0_64.0_Nxz_128_64",
    AMD128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_AMD_Lxz_128.0_64.0_Nxz_256_128",

    W032 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_128.0_64.0_Nxz_64_32",
    W064 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_128.0_64.0_Nxz_128_64",
    W0128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu0_Lxz_128.0_64.0_Nxz_256_128",

    W532 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_128.0_64.0_Nxz_64_32",
    W564 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_128.0_64.0_Nxz_128_64",
    W5128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9nu1e-5_Lxz_128.0_64.0_Nxz_256_128",

    WA32 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_128.0_64.0_Nxz_64_32",
    WA64 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_128.0_64.0_Nxz_128_64",
    WA128 = "linearb_dbdz_9.0e-5_QU_0.0_QB_4.24e-8_f_0.0001_WENO9AMD_Lxz_128.0_64.0_Nxz_256_128"
)

bbar_datas_2D = NamedTuple{keys(filenames_2D)}([FieldTimeSeries("./LES/$(value)/instantaneous_timeseries.jld2", "bbar") for value in values(filenames_2D)])
bbar_datas_3D = NamedTuple{keys(filenames_3D)}([FieldTimeSeries("./LES/$(value)/instantaneous_timeseries.jld2", "bbar") for value in values(filenames_3D)])

Nzs_2D = NamedTuple{keys(bbar_datas_2D)}([data.grid.Nz for data in bbar_datas_2D])
Nzs_3D = NamedTuple{keys(bbar_datas_3D)}([data.grid.Nz for data in bbar_datas_3D])

zCs_2D = NamedTuple{keys(bbar_datas_2D)}([data.grid.zᵃᵃᶜ[1:Nz] for (Nz, data) in zip(values(Nzs_2D), values(bbar_datas_2D))])
zCs_3D = NamedTuple{keys(bbar_datas_3D)}([data.grid.zᵃᵃᶜ[1:Nz] for (Nz, data) in zip(values(Nzs_3D), values(bbar_datas_3D))])

ts = bbar_datas_2D[1].times
Nt = length(ts)

Qᴮ, N², f₀ = jldopen("./LES/$(filenames_2D[1])/instantaneous_timeseries.jld2", "r") do file
    file["metadata/buoyancy_flux"], file["metadata/buoyancy_gradient"], file["metadata/coriolis_parameter"]
end

#%%
fig = Figure(size=(2400, 600))
axSL = CairoMakie.Axis(fig[1, 1], xlabel="b (m s⁻²)", ylabel="z (m)", title="Smagorinsky-Lilly")
axAMD = CairoMakie.Axis(fig[1, 2], xlabel="b (m s⁻²)", ylabel="z (m)", title="AMD")
axW0 = CairoMakie.Axis(fig[1, 3], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 0 m² s⁻¹")
axW1 = CairoMakie.Axis(fig[1, 4], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 1e-5 m² s⁻¹")
axWA = CairoMakie.Axis(fig[1, 5], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9) + AMD")

n = Observable(1)

b_SL32ₙ = @lift interior(bbar_datas_2D.SL32[$n], 1, 1, :)
b_SL64ₙ = @lift interior(bbar_datas_2D.SL64[$n], 1, 1, :)
b_SL128ₙ = @lift interior(bbar_datas_2D.SL128[$n], 1, 1, :)

b_AMD32ₙ = @lift interior(bbar_datas_2D.AMD32[$n], 1, 1, :)
b_AMD64ₙ = @lift interior(bbar_datas_2D.AMD64[$n], 1, 1, :)
b_AMD128ₙ = @lift interior(bbar_datas_2D.AMD128[$n], 1, 1, :)

b_W032ₙ = @lift interior(bbar_datas_2D.W032[$n], 1, 1, :)
b_W064ₙ = @lift interior(bbar_datas_2D.W064[$n], 1, 1, :)
b_W0128ₙ = @lift interior(bbar_datas_2D.W0128[$n], 1, 1, :)

b_W532ₙ = @lift interior(bbar_datas_2D.W532[$n], 1, 1, :)
b_W564ₙ = @lift interior(bbar_datas_2D.W564[$n], 1, 1, :)
b_W5128ₙ = @lift interior(bbar_datas_2D.W5128[$n], 1, 1, :)

b_WA32ₙ = @lift interior(bbar_datas_2D.WA32[$n], 1, 1, :)
b_WA64ₙ = @lift interior(bbar_datas_2D.WA64[$n], 1, 1, :)
b_WA128ₙ = @lift interior(bbar_datas_2D.WA128[$n], 1, 1, :)

lines!(axSL, b_SL32ₙ, zCs_2D.SL32, label="2m resolution")
lines!(axSL, b_SL64ₙ, zCs_2D.SL64, label="1m resolution")
lines!(axSL, b_SL128ₙ, zCs_2D.SL128, label="0.5m resolution")

lines!(axAMD, b_AMD32ₙ, zCs_2D.AMD32, label="2m resolution")
lines!(axAMD, b_AMD64ₙ, zCs_2D.AMD64, label="1m resolution")
lines!(axAMD, b_AMD128ₙ, zCs_2D.AMD128, label="0.5m resolution")

lines!(axW0, b_W032ₙ, zCs_2D.W032, label="2m resolution")
lines!(axW0, b_W064ₙ, zCs_2D.W064, label="1m resolution")
lines!(axW0, b_W0128ₙ, zCs_2D.W0128, label="0.5m resolution")

lines!(axW1, b_W532ₙ, zCs_2D.W532, label="2m resolution")
lines!(axW1, b_W564ₙ, zCs_2D.W564, label="1m resolution")
lines!(axW1, b_W5128ₙ, zCs_2D.W5128, label="0.5m resolution")

lines!(axWA, b_WA32ₙ, zCs_2D.WA32, label="2m resolution")
lines!(axWA, b_WA64ₙ, zCs_2D.WA64, label="1m resolution")
lines!(axWA, b_WA128ₙ, zCs_2D.WA128, label="0.5m resolution")

axislegend(axSL, position=:rb)

title = @lift "2D LES, Qᴮ = $(Qᴮ) m² s⁻³, N² = $(N²) s⁻², f = $(f₀) s⁻¹, $(round(ts[$n] / 24 / 60^2, digits=2)) days"

Label(fig[0, :], title, font=:bold, tellwidth=false)

display(fig)

CairoMakie.record(fig, "./Data/2D_LES_resolution_closure_QB_$(Qᴮ)_N2_$(N²)_f_$(f₀).mp4", 1:Nt, framerate=40) do nn
    n[] = nn
end
#%%
fig = Figure(size=(2400, 600))
axSL = CairoMakie.Axis(fig[1, 1], xlabel="b (m s⁻²)", ylabel="z (m)", title="Smagorinsky-Lilly")
axAMD = CairoMakie.Axis(fig[1, 2], xlabel="b (m s⁻²)", ylabel="z (m)", title="AMD")
axW0 = CairoMakie.Axis(fig[1, 3], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 0 m² s⁻¹")
axW1 = CairoMakie.Axis(fig[1, 4], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 1e-5 m² s⁻¹")
axWA = CairoMakie.Axis(fig[1, 5], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9) + AMD")

n = Observable(1)

b_SL32ₙ = @lift interior(bbar_datas_3D.SL32[$n], 1, 1, :)
b_SL64ₙ = @lift interior(bbar_datas_3D.SL64[$n], 1, 1, :)
b_SL128ₙ = @lift interior(bbar_datas_3D.SL128[$n], 1, 1, :)

b_AMD32ₙ = @lift interior(bbar_datas_3D.AMD32[$n], 1, 1, :)
b_AMD64ₙ = @lift interior(bbar_datas_3D.AMD64[$n], 1, 1, :)
b_AMD128ₙ = @lift interior(bbar_datas_3D.AMD128[$n], 1, 1, :)

b_W032ₙ = @lift interior(bbar_datas_3D.W032[$n], 1, 1, :)
b_W064ₙ = @lift interior(bbar_datas_3D.W064[$n], 1, 1, :)
b_W0128ₙ = @lift interior(bbar_datas_3D.W0128[$n], 1, 1, :)

b_W532ₙ = @lift interior(bbar_datas_3D.W532[$n], 1, 1, :)
b_W564ₙ = @lift interior(bbar_datas_3D.W564[$n], 1, 1, :)
b_W5128ₙ = @lift interior(bbar_datas_3D.W5128[$n], 1, 1, :)

b_WA32ₙ = @lift interior(bbar_datas_3D.WA32[$n], 1, 1, :)
b_WA64ₙ = @lift interior(bbar_datas_3D.WA64[$n], 1, 1, :)
b_WA128ₙ = @lift interior(bbar_datas_3D.WA128[$n], 1, 1, :)

lines!(axSL, b_SL32ₙ, zCs_3D.SL32, label="2m resolution")
lines!(axSL, b_SL64ₙ, zCs_3D.SL64, label="1m resolution")
lines!(axSL, b_SL128ₙ, zCs_3D.SL128, label="0.5m resolution")

lines!(axAMD, b_AMD32ₙ, zCs_3D.AMD32, label="2m resolution")
lines!(axAMD, b_AMD64ₙ, zCs_3D.AMD64, label="1m resolution")
lines!(axAMD, b_AMD128ₙ, zCs_3D.AMD128, label="0.5m resolution")

lines!(axW0, b_W032ₙ, zCs_3D.W032, label="2m resolution")
lines!(axW0, b_W064ₙ, zCs_3D.W064, label="1m resolution")
lines!(axW0, b_W0128ₙ, zCs_3D.W0128, label="0.5m resolution")

lines!(axW1, b_W532ₙ, zCs_3D.W532, label="2m resolution")
lines!(axW1, b_W564ₙ, zCs_3D.W564, label="1m resolution")
lines!(axW1, b_W5128ₙ, zCs_3D.W5128, label="0.5m resolution")

lines!(axWA, b_WA32ₙ, zCs_3D.WA32, label="2m resolution")
lines!(axWA, b_WA64ₙ, zCs_3D.WA64, label="1m resolution")
lines!(axWA, b_WA128ₙ, zCs_3D.WA128, label="0.5m resolution")

axislegend(axSL, position=:rb)

title = @lift "3D LES, Qᴮ = $(Qᴮ) m² s⁻³, N² = $(N²) s⁻², f = $(f₀) s⁻¹, $(round(ts[$n] / 24 / 60^2, digits=2)) days"

Label(fig[0, :], title, font=:bold, tellwidth=false)

display(fig)

CairoMakie.record(fig, "./Data/3D_LES_resolution_closure_QB_$(Qᴮ)_N2_$(N²)_f_$(f₀).mp4", 1:Nt, framerate=40) do nn
    n[] = nn
end
#%%
fig = Figure(size=(2400, 600))
axSL = CairoMakie.Axis(fig[1, 1], xlabel="b (m s⁻²)", ylabel="z (m)", title="Smagorinsky-Lilly")
axAMD = CairoMakie.Axis(fig[1, 2], xlabel="b (m s⁻²)", ylabel="z (m)", title="AMD")
axW0 = CairoMakie.Axis(fig[1, 3], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 0 m² s⁻¹")
axW1 = CairoMakie.Axis(fig[1, 4], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9), ν = κ = 1e-5 m² s⁻¹")
axWA = CairoMakie.Axis(fig[1, 5], xlabel="b (m s⁻²)", ylabel="z (m)", title="WENO(9) + AMD")

n = Observable(1)

b_2D_SL128ₙ = @lift interior(bbar_datas_2D.SL128[$n], 1, 1, :)
b_2D_AMD128ₙ = @lift interior(bbar_datas_2D.AMD128[$n], 1, 1, :)
b_2D_W0128ₙ = @lift interior(bbar_datas_2D.W0128[$n], 1, 1, :)
b_2D_W5128ₙ = @lift interior(bbar_datas_2D.W5128[$n], 1, 1, :)
b_2D_WA128ₙ = @lift interior(bbar_datas_2D.WA128[$n], 1, 1, :)

b_3D_SL128ₙ = @lift interior(bbar_datas_3D.SL128[$n], 1, 1, :)
b_3D_AMD128ₙ = @lift interior(bbar_datas_3D.AMD128[$n], 1, 1, :)
b_3D_W0128ₙ = @lift interior(bbar_datas_3D.W0128[$n], 1, 1, :)
b_3D_W5128ₙ = @lift interior(bbar_datas_3D.W5128[$n], 1, 1, :)
b_3D_WA128ₙ = @lift interior(bbar_datas_3D.WA128[$n], 1, 1, :)

lines!(axSL, b_2D_SL128ₙ, zCs_2D.SL128, label="2D")
lines!(axAMD, b_2D_AMD128ₙ, zCs_2D.AMD128, label="2D")
lines!(axW0, b_2D_W0128ₙ, zCs_2D.W0128, label="2D")
lines!(axW1, b_2D_W5128ₙ, zCs_2D.W5128, label="2D")
lines!(axWA, b_2D_WA128ₙ, zCs_2D.WA128, label="2D")

lines!(axSL, b_3D_SL128ₙ, zCs_3D.SL128, label="3D")
lines!(axAMD, b_3D_AMD128ₙ, zCs_3D.AMD128, label="3D")
lines!(axW0, b_3D_W0128ₙ, zCs_3D.W0128, label="3D")
lines!(axW1, b_3D_W5128ₙ, zCs_3D.W5128, label="3D")
lines!(axWA, b_3D_WA128ₙ, zCs_3D.WA128, label="3D")

axislegend(axSL, position=:rb)

title = @lift "2D vs 3D LES, 0.5m resolution, Qᴮ = $(Qᴮ) m² s⁻³, N² = $(N²) s⁻², f = $(f₀) s⁻¹, $(round(ts[$n] / 24 / 60^2, digits=2)) days"

Label(fig[0, :], title, font=:bold, tellwidth=false)

display(fig)

CairoMakie.record(fig, "./Data/2D_3D_LES_0.5mresolution_closure_QB_$(Qᴮ)_N2_$(N²)_f_$(f₀).mp4", 1:Nt, framerate=40) do nn
    n[] = nn
end
#%%