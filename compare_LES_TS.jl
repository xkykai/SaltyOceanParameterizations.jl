using CairoMakie
using Oceananigans
using JLD2

FILE_DIRS = [
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_64.0_Ly_64.0_Nz_128_Nx_32_Ny_32",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_64.0_Ly_64.0_Nz_128_Nx_32_Ny_32",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_64.0_Ly_64.0_Nz_256_Nx_64_Ny_64",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_64.0_Ly_64.0_Nz_256_Nx_64_Ny_64",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_64.0_Ly_64.0_Nz_512_Nx_128_Ny_128",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_64.0_Ly_64.0_Nz_512_Nx_128_Ny_128"
    # "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_128.0_Ly_128.0_Nz_128_Nx_64_Ny_64",
    # "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_128.0_Ly_128.0_Nz_128_Nx_64_Ny_64",
    # "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_128.0_Ly_128.0_Nz_256_Nx_128_Ny_128",
    # "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_20.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_128.0_Ly_128.0_Nz_256_Nx_128_Ny_128",
]

labels = [
    "WENO(9), ν = κ = 1e-5, 2m resolution",
    "AMD, 2m resolution",
    "WENO(9), ν = κ = 1e-5, 1m resolution",
    "AMD, 1m resolution",
    "WENO(9), ν = κ = 1e-5, 0.5m resolution",
    "AMD, 0.5m resolution"
]

T_top = 20.
S_top = 35.

video_name = "./Data/QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_Ttop_$(T_top)_Stop_$(S_top)_TS_resolution.mp4"

T_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar") for FILE_DIR in FILE_DIRS]
S_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar") for FILE_DIR in FILE_DIRS]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᵁ = parameters["momentum_flux"]
Qᵀ = parameters["temperature_flux"]
Qˢ = parameters["salinity_flux"]
# T_top = parameters["surface_temperature"]
# S_top = parameters["surface_salinity"]

Nxs = [size(data.grid)[1] for data in T_datas]
Nys = [size(data.grid)[2] for data in T_datas]
Nzs = [size(data.grid)[3] for data in T_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(T_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(T_datas)]

#%%
fig = Figure(resolution = (2200, 1200))
axT = Axis(fig[1, 1], title="T (°C)", ylabel="z")
axS = Axis(fig[1, 2], title="S (g/kg)", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

Tlim = (find_min(T_datas...), find_max(T_datas...))
Slim = (find_min(S_datas...), find_max(S_datas...))

n = Observable(1)

times = u_datas[1].times
Nt = length(times)
time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

Tₙs = [@lift interior(data[$n], 1, 1, :) for data in T_datas]
Sₙs = [@lift interior(data[$n], 1, 1, :) for data in S_datas]

for i in 1:length(FILE_DIRS)
    lines!(axT, Tₙs[i], zCs[i], label=labels[i])
    lines!(axS, Sₙs[i], zCs[i], label=labels[i])
end

# make a legend
Legend(fig[2, :], axT, tellwidth=false, orientation=:horizontal)

xlims!(axT, Tlim)
xlims!(axS, Slim)

trim!(fig.layout)

record(fig, video_name, 1:Nt, framerate=30) do nn
    n[] = nn
end

@info "Animation complete"
#%%